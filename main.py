import argparse
import tqdm
import random
import math
import os
import pandas as pd

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
import torch.optim as optim
from classifier import *
import visdom
vis = visdom.Visdom()
draw_graph = None
draw_accuracy = None

csv_file = "classifications.csv"
root_dir = "data/"
batch_size = 159
learning_rate = 0.001
epoch_num = 30
save_model = True
experiment_num = 6

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = AsteroidDataset(csv_file=csv_file, root_dir=root_dir, train=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

validation_dataset = AsteroidDataset(csv_file=csv_file, root_dir=root_dir, train=False, transform=transform)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1)


classifier = CNN()

criterion = nn.NLLLoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

def model_save(model, path):
    pickle.dump(model, open(path, 'wb'))

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

print('Starting training...')
total_iter = 0

for epoch in range(epoch_num):
    corrects = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs = data["image"]
        labels = data["class"]

        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        output = classifier(inputs)
        loss = criterion(output, labels)
        update = None if draw_graph is None else 'append'
        draw_graph = vis.line(X = np.array([total_iter]), Y = np.array([loss.data[0]]), win = draw_graph, update = update, opts=dict(title="NLL loss"))


        temp = output[:, 0].data.numpy()
        temp = np.apply_along_axis(lambda x: np.ceil(np.exp(x)), 0, temp)
        temp = torch.from_numpy(temp).long()
        accuracy = torch.sum(temp == labels.data)/ float(batch_size)       

        update = None if draw_accuracy is None else 'append'
        draw_accuracy = vis.line(X = np.array([total_iter]), Y = np.array([accuracy]), win = draw_accuracy, update = update, opts=dict(title="Accuracy"))
        
        print("[EPOCH %d ITER %d] Loss: %f (accuracy: %f)" % (epoch, i, loss.data[0], accuracy))

        loss.backward()
        optimizer.step()
        total_iter += 1
    adjust_learning_rate(optimizer, epoch)

if save_model:
    model_save(classifier, "saved_models/experiment_"+str(experiment_num))


