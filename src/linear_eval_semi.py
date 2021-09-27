import os
import torch
import torch.nn as nn
import argparse
import urllib
import pytorch_lightning as pl
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.optim import SGD
from torch.nn import functional as F
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import LearningRateMonitor

from src.optimizers import linear_warmup_decay
from src.models import resnet18, resnet50, resnet50w2, resnet50w4
from src.datamodules import cifar10_normalization, stl10_normalization, imagenet_normalization

from typing import Union, List, Optional, Sequence, Dict, Iterator, Tuple


ENCODERS = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet50w2": resnet50w2,
    "resnet50w4": resnet50w4,
}


class LinearEvaluation(pl.LightningModule):

    def __init__(
        self,
        encoder: nn.Module,
        encoder_output_dim: int,
        num_classes: int,
        num_samples: int,
        batch_size: int,
        gpus: int,
        max_epochs: int,
        learning_rate: float,
        learning_rate_last_layer: float,
        weight_decay: float,
        nesterov: bool,
        momentum: float,
        decay_epochs: list,
        gamma: float,
    ) -> None:

        super().__init__()
        assert isinstance(encoder, nn.Module)

        self.encoder = encoder
        self.linear_layer = nn.Linear(encoder_output_dim, num_classes)

        self.batch_size = batch_size
        self.num_samples = num_samples
        self.gpus = gpus

        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.learning_rate_last_layer = learning_rate_last_layer
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.momentum = momentum
        self.decay_epochs = decay_epochs
        self.gamma = gamma

        global_batch_size = (
            self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        )
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        # metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.test_acc = Accuracy(compute_on_step=False)

    def on_train_epoch_start(self) -> None:
        self.encoder.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            [
                {
                    'params': self.encoder.parameters()
                },
                {
                    'params': self.linear_layer.parameters(),
                    'lr': self.learning_rate_last_layer,
                }
            ],
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.decay_epochs, gamma=self.gamma
        )

        return [optimizer], [scheduler]

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
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument("--labels_percentage", type=str, default="10", choices=["1", "10"])
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--input_height", type=int, default=32)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt')
    parser.add_argument("--data_path", type=str, default=".")

    parser.add_argument("--batch_size", default=256, type=int, help="batch size per gpu")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
    parser.add_argument("--gpus", default=1, type=int, help="number of GPUs")
    parser.add_argument('--max_epochs', default=20, type=int, help="number of epochs")

    # fine-tuner params
    parser.add_argument("--learning_rate_last_layer", type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.2, help="lr decay factor")
    parser.add_argument("--decay_epochs", type=int, nargs="+", default=[12, 16],
                        help="Epochs at which to decay learning rate.")

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    args.learning_rate = 0.1 * int(args.batch_size / 256)

    # set hidden dim for resnet18
    if args.encoder_name == "resnet18":
        args.encoder_output_dim = 512

    # initialize data sources
    train_data_path = os.path.join(args.data_path, "train")
    train_dataset = datasets.ImageFolder(train_data_path)

    """
    snippet taken from: https://github.com/facebookresearch/swav/blob/main/eval_semisup.py
    """
    # take either 1% or 10% of images
    subset_file = urllib.request.urlopen(
        "https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/"
        + str(args.labels_percentage)
        + "percent.txt"
    )
    list_imgs = [li.decode("utf-8").split('\n')[0] for li in subset_file]
    train_dataset.samples = [(
        os.path.join(train_data_path, li.split('_')[0], li),
        train_dataset.class_to_idx[li.split('_')[0]]
    ) for li in list_imgs]

    val_dataset = datasets.ImageFolder(os.path.join(args.data_path, "val"))

    args.num_samples = len(train_dataset.samples)
    args.input_height = 224

    args.first_conv3x3 = False # first conv is 7x7 for imagenet
    args.remove_first_maxpool = False # don't remove first maxpool
    normalization = imagenet_normalization()

    train_dataset.transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalization
    ])

    val_dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalization
    ])

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

    encoder = ENCODERS[args.encoder_name](first_conv3x3=args.first_conv3x3, remove_first_maxpool=args.remove_first_maxpool)

    # load encoder weights from ckpt
    device = torch.device(encoder.conv1.weight.device)
    ckpt_model = torch.load(args.ckpt_path, map_location=device)
    encoder_dict = {}

    for k in ckpt_model['state_dict'].keys():
        if 'encoder' in k:
            encoder_dict[k.replace('encoder.', '')] = ckpt_model['state_dict'][k]
    encoder.load_state_dict(encoder_dict, strict=True)

    linear_eval = LinearEvaluation(
        encoder=encoder,
        encoder_output_dim=args.encoder_output_dim,
        num_classes=args.num_classes,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        learning_rate_last_layer=args.learning_rate_last_layer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        momentum=args.momentum,
        decay_epochs=args.decay_epochs,
        gamma=args.gamma,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        distributed_backend="ddp" if args.gpus > 1 else None,
        precision=16,
        callbacks=[LearningRateMonitor(logging_interval="step")]
    )

    trainer.fit(linear_eval, train_loader)
    trainer.test(test_dataloaders=val_loader)
