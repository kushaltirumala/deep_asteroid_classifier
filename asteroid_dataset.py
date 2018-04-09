from skimage import io
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import numpy as np

class AsteroidDataset(Dataset):

    def __init__(self, csv_file, root_dir, train=True, transform=None):
        temp = pd.read_csv(csv_file)
        self.df = temp[0:4*int(len(temp)/5)] if train else temp[4*int(len(temp)/5): len(temp)]
        self.root_dir = root_dir
        self.transform = transform
        self.col_idx = list(self.df.columns).index("links")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, self.col_idx])
        image = np.array(Image.open(img_name).resize((150,150), Image.BILINEAR))
        # type of image is numpy ndarray
        if self.transform:
            image = self.transform(image)

        classification = self.df.iloc[idx]["classification"]

        sample = {'image': image, 'class': classification}

        return sample