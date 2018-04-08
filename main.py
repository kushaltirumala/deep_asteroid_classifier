import random
import math
import os


import argparse
import tqdm

import numpy as np
import struct

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pickle
import torch
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from asteroid_dataset import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = AsteroidDataset(csv_file="classifications.csv", root_dir="data/", train=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True, num_workers=1)

validation_dataset = AsteroidDataset(csv_file="classifications.csv", root_dir="data/", train=False, transform=transform)
validation_dataloader = DataLoader(validation_dataset, batch_size=30, shuffle=True, num_workers=1)











