from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        cwd = os.getcwd()
        img_path = os.path.join(cwd, self.img_dir)
        os.chdir(img_path)
        image = Image.open(os.path.join(img_path, self.img_labels.iloc[idx, 0]))
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        os.chdir(cwd)
        return image, label
