from skimage import io
from torch.utils.data import Dataset
import pandas as pd

class AsteroidDataset(Dataset):

    def __init__(self, csv_file, root_dir, train=True, transform=None):
        temp = pd.read_csv(csv_file)
        self.df = temp[0:4*int(len(temp)/5)] if train else temp[4*int(len(temp)/5): len(temp)]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 1])
        image = io.imread(img_name)
        classification = self.df.iloc[idx]["classification"]
        sample = {'image': image, 'class': classification}

        if self.transform:
            sample = self.transform(sample)

        return sample