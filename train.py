import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm, trange
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
# %matplotlib inline
import matplotlib.pyplot as plt

from Pointnet_model import Pointnet, loss_function
from modelnet40_data import PointnetDataset, load_data, PointSampler, Normalize, RandomNoise, RandomRotate, toTensor
from Pointnet_train_test import test, run

from visualization import pcshow

import copy
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import torch.nn.functional as F

if torch.cuda.is_available():
      device = torch.device('cuda:0')
      print('running on GPU')
else:
      device = torch.device('cpu')
      print('running on CPU')

classes = {'airplane' : 0, 'bathtub' : 1, 'bed': 2, 'bench': 3, 'bookshelf':4, 'bottle':5, 'bowl' : 6, 'car':7, 'chair' : 8, 'cone' : 9, 'cup':10, 'curtain':11 , 'desk':12, 'door':13, 'dresser':14, 'flower_pot':15, 'glass_box':16, 'guitar':17, 'keyboard':18, 'lamp':19, 'laptop':20, 'mantel':21, 'monitor':22, 'nightstand':23, 'person':24, 'piano':25, 'plant':26, 'radio':27, 'range_hood':28, 'sink':29, 'sofa':30, 'stairs':31, 'stool':32, 'table':33, 'tent':34, 'toilet':35, 'tv_stand':36, 'vase':37, 'wardrobe':38, 'xbox' : 39}
res_classes = dict((v,k) for k,v in classes.items())

def random_noise(verts):
  noise = np.random.normal(0, 0.05, verts.shape)
  noise = np.clip(noise, -0.1, 0.1)
  return verts + noise

default_transform = transforms.Compose(
        [
            PointSampler(1024),
            Normalize(),
            toTensor()]
    )
train_transform = transforms.Compose(
        [
            PointSampler(1024),
            Normalize(),
            RandomNoise(),
            RandomRotate(),
            toTensor()]
    )
train_X, train_y = load_data()
test_X, test_y = load_data(mode='test')
testset = PointnetDataset(test_X, test_y, transform=default_transform)
trainset = PointnetDataset(train_X, train_y, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)
model = Pointnet(device=device).to(device)
n_epochs = 250

train_hist, test_hist = run(model, n_epochs, device, trainloader, testloader)
torch.save({
            'model_state_dict': model.state_dict()}, 'new_model.pt')
