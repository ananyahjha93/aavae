import os

from torchvision.io import read_image
from torch.utils.data import Dataset


class Places205(Dataset):
    def __init__(self, img_dir, split='train', transform=None):
        super().__init__()

        self.img_dir = img_dir
        self.split = split
        self.transform = transform

        # load 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])

        image = read_image(img_path)
        label = self.img_labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, self.images[i][1]
