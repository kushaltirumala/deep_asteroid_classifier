import os
from skimage import io

class AsteroidDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
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