import torch
import torch.nn as nn
import argparse
import pytorch_lightning as pl
import torchvision.transforms as transforms

from torch.optim import SGD
from torch.nn import functional as F
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import LearningRateMonitor

from src.optimizers import linear_warmup_decay
from src.models import resnet18, resnet50, resnet50w2, resnet50w4

from places205 import Places205, places205_normalization

from typing import Union, List, Optional, Sequence, Dict, Iterator, Tuple


ENCODERS = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet50w2": resnet50w2,
    "resnet50w4": resnet50w4,
}


class DownstreamClassificationEval(pl.LightningModule):

    def __init__(
        self,
        encoder: nn.Module,
        encoder_output_dim: int,
        num_classes: int,
        max_epochs: int,
        learning_rate: float,
        milestones: list,
    ) -> None:

        super().__init__()
        assert isinstance(encoder, nn.Module)

        self.encoder = encoder
        self.linear_layer = nn.Linear(encoder_output_dim, num_classes)

        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.milestones = milestones

        # metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.test_acc = Accuracy(compute_on_step=False)

    def on_train_epoch_start(self) -> None:
        self.encoder.eval()

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.linear_layer.parameters(),
        #     lr=self.learning_rate,
        #     momentum=0.9,
        #     weight_decay=5e-4,
        # )

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=self.milestones, gamma=0.1
        # )

        # return [optimizer], [scheduler]
        optimizer = torch.optim.Adam(self.linear_layer.parameters(), lr=1e-4)
        return optimizer

    def shared_step(self, batch):
        x, y = batch

        with torch.no_grad():
            feats = self.encoder(x)

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        loss = F.cross_entropy(logits, y)

        return loss, logits, y

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, y = self.shared_step(batch)
        acc = self.train_acc(F.softmax(logits, dim=1), y)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_acc_step', acc, on_step=True, on_epoch=False)
        self.log('train_acc_epoch', self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, y = self.shared_step(batch)
        self.val_acc(F.softmax(logits, dim=1), y)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, y = self.shared_step(batch)
        self.test_acc(F.softmax(logits, dim=1), y)

        self.log('test_loss', loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)

        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)  # added to launch 2 ddp script on same node

    # encoder params
    parser.add_argument("--encoder_name", default="resnet50", choices=ENCODERS.keys())
    parser.add_argument('--encoder_output_dim', type=int, default=2048)
    parser.add_argument("--first_conv3x3", type=bool, default=True)  # default for cifar-10
    parser.add_argument("--remove_first_maxpool", type=bool, default=True)  # default for cifar-10

    # eval params
    parser.add_argument('--dataset', type=str, help='inat18, places205', default='places205')
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt')
    parser.add_argument("--data_path", type=str, default=".")

    parser.add_argument("--batch_size", default=256, type=int, help="batch size per gpu")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
    parser.add_argument("--gpus", default=1, type=int, help="number of GPUs")
    parser.add_argument('--max_epochs', default=90, type=int, help="number of epochs")

    # fine-tuner params
    parser.add_argument('--learning_rate', type=float, default=0.01)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # set hidden dim for resnet18
    if args.encoder_name == "resnet18":
        args.encoder_output_dim = 512

    # initialize datamodules
    train_transforms = None
    eval_transforms = None

    if args.dataset == 'places205':
        args.max_epochs = 14
        args.milestones = [5, 10]

        num_classes = 205
        args.first_conv3x3 = False # first conv is 7x7 for places205
        args.remove_first_maxpool = False # don't remove first maxpool
        normalization = places205_normalization()

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalization
        ])

        train_dataset = Places205(img_dir=args.data_path, split='train', transform=train_transform)
        val_dataset = Places205(img_dir=args.data_path, split='val', transform=val_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    encoder = ENCODERS[args.encoder_name](first_conv3x3=args.first_conv3x3, remove_first_maxpool=args.remove_first_maxpool)

    # load encoder weights from ckpt
    device = torch.device(encoder.conv1.weight.device)
    ckpt_model = torch.load(args.ckpt_path, map_location=device)
    encoder_dict = {}

    for k in ckpt_model['state_dict'].keys():
        if 'encoder' in k:
            encoder_dict[k.replace('encoder.', '')] = ckpt_model['state_dict'][k]
    encoder.load_state_dict(encoder_dict, strict=True)

    classification_task = DownstreamClassificationEval(
        encoder=encoder,
        encoder_output_dim=args.encoder_output_dim,
        num_classes=num_classes,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        milestones=args.milestones,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        distributed_backend="ddp" if args.gpus > 1 else None,
        precision=16,
        callbacks=[LearningRateMonitor(logging_interval="step")],
    )

    trainer.fit(classification_task, train_loader)
    trainer.test(test_dataloaders=val_loader)
