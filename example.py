import os
import pickle
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import sys



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def load_model(path):
    print ("Loading learned model")
    discriminator = pickle.load(open(path, 'rb'))
    return discriminator

def embed_input(path, model):
    image = np.array(Image.open(path).resize((150,150), Image.BILINEAR).crop((0,0,150,135)))
    image = transform(image)
    image = Variable(image)
    image = image.unsqueeze(0)
    output = model(image)
    temp = output[:, 1].data.numpy()
    temp = np.apply_along_axis(lambda x: np.rint(np.exp(x)), 0, temp)
    return temp[0]


if __name__ == "__main__":
    path = sys.argv[1]
    model = load_model("saved_models/experiment_14")

    if os.path.isfile(path):
        print("detected only one file")
        classification = embed_input(path, model)
        print(classification)
    elif os.path.isdir(path):
        print("detected directory of files")