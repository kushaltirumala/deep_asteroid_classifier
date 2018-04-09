import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.math_utils import *

class CNN(nn.Module):
	def __init__(self, num_layers=2, hidden_num=32):
		super(CN, self).__init__()
		self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(32*32*34, 2)

    def forward(self, x):
    	x = self.layer1(x)
    	x = self.layer2(x)
    	x = x.view(-1, 32*32*34)
    	x = fc(x)
    	return F.softmax(x, dim=1)

	