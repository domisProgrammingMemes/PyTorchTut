# Last "60min Blitz Tutorial - Training a Classifier: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# Train a classifier on the CIFAR10 dataset

# imports
import torch                                            # used for: to use pytorch in general (e.g. Tensors)
import torchvision                                      # used for: to work with data for nn
import torchvision.transforms as transforms             # used for: to transform data in our datasets
import torchvision.datasets as datasets                 # used for: has standard datasets we can import in an easy way

from torch.utils.data import DataLoader                 # Gives easier dataset management and creates mini-batches

import torch.nn as nn                                   # used for: All neural network modules (nn.Linear, nn.Conv2d)
import torch.nn.functional as F                         # used for: All functions that don't have any parameters
import torch.optim as optim                             # used for: For optimization algorithms (SGD, ADAM)

import matplotlib.pyplot as plt                         # used for: for plotting images
import numpy as np                                      # used for: calculations with matrix data


# Path to save and load model
net_path = './CIFAR_net.pth'
# Path for Data
data_path = './data'

# set up the divice (GPU or CPU) via input prompt
cuda_true = input("Use GPU? (y) or (n)?")
if cuda_true == "y":
    device = "cuda"
else:
    device = "cpu"
print("Device:", device)

# Hyperparameters
num_epochs = 10
train_batch_size = 64
test_batch_size = 64
mini_batch_size = 100
learning_rate = 0.001
momentum = 0.5

# Normalization on the pictures
normalize = transforms.Normalize(mean=0.5, std=1)
transform = transforms.transforms.Compose(
    [transforms.ToTensor(),
     normalize]
)

# Start Tutorial
# Generally use python packages that load data into a numpy array -> then convert to torch.Tensor!
# -> for images use e.g. Pillow, OpenCV

# For vision PyTorch created torchvision which has:
# dataloaders for common datasets (MNIST, CIFAR10) and
# datatransformers for images, viz. 'torchvision.datasets' and 'torch.utils.data.DataLoader'

# This tutorial focuses on CIFAR10 which has 10 classes ('airplane', 'automobile', ...)
# Images are of size 3x32x32, i.e. 3-channel color images of 32x32 pixel

