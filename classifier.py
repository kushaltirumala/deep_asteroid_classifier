import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.math_utils import *

class Classifier(nn.Module):
	def __init__(self, num_layers=2, hidden_num=32):
		
	