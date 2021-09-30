import os
import pandas
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


def places205_normalization():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize


class Places205(Dataset):
    train_metadata = 'train_places205.csv'
    val_metadata = 'val_places205.csv'

    def __init__(self, img_dir, split='train', transform=None):
        super().__init__()

        self.img_dir = img_dir
        self.split = split
        self.transform = transform

        if split == 'train':
            metadata_path = os.path.join(self.img_dir, self.train_metadata)
        else:
            metadata_path = os.path.join(self.img_dir, self.val_metadata)

        self.metadata = pandas.read_csv(metadata_path, sep=' ', header=None)
        self.images = self.metadata.iloc[:, 0].values
        self.labels = self.metadata.iloc[:, 1].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])

        img = Image.open(img_path)
        img = img.convert('RGB')
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
