# Last "60min Blitz Tutorial - Training a Classifier: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# Train a classifier on the CIFAR10 dataset

# imports
import torch  # used for: to use pytorch in general (e.g. Tensors)
import torchvision  # used for: to work with data for nn
import torchvision.transforms as transforms  # used for: to transform data in our datasets
import torchvision.datasets as datasets  # used for: has standard datasets we can import in an easy way

from torch.utils.data import DataLoader  # Gives easier dataset management and creates mini-batches

import torch.nn as nn  # used for: All neural network modules (nn.Linear, nn.Conv2d)
import torch.nn.functional as F  # used for: All functions that don't have any parameters
import torch.optim as optim  # used for: For optimization algorithms (SGD, ADAM)

import matplotlib.pyplot as plt  # used for: for plotting images
import numpy as np  # used for: calculations with matrix data

# ----------------------------------------------------------------------------------------------------

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
learning_rate = 0.001
momentum = 0.5

# Normalization on the pictures
normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # mean and std
transform = transforms.transforms.Compose(
    [transforms.ToTensor(),
     normalize]
)

# ----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Start Tutorial
    # Generally use python packages that load data into a numpy array -> then convert to torch.Tensor!
    # -> for images use e.g. Pillow, OpenCV

    # For vision PyTorch created torchvision which has:
    # dataloaders for common datasets (MNIST, CIFAR10) and
    # datatransformers for images, viz. 'torchvision.datasets' and 'torch.utils.data.DataLoader'

    # This tutorial focuses on CIFAR10 which has 10 classes ('airplane', 'automobile', ...)
    # Images are of size 3x32x32, i.e. 3-channel color images of 32x32 pixel

    # Steps (in order):
    # 1. Load and normalizing the CIFAR10 training and test datasets using torchvision
    # 2. Define a Convolutional Neural Network
    # 3. Define a loss function
    # 4. Train the network on the training data
    # 5. Test the network on the test data

    # ----------------------------------------------------------------------------------------------------

    # step 1:
    # the output of torchvision datasets are PILImages of range [0, 1].
    # transform to Tensors of normalized range [-1, 1]

    trainset = datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
    trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=0)

    testset = datasets.CIFAR10(root=data_path, train=False, transform=transform, download=True)
    testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=0)

    # classes from dataset (this is a tuple:https://www.w3schools.com/python/python_tuples.asp):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # show an example picture
    def example():
        # no step, just show pictures with pyplot and numpy
        # functions to show an image
        def showpicture(img):
            img = img / 2 + 0.5                                     # unnormalize [remember: range is [-1, 1] | (e.g. pixel(14, 14) = 1 -> 1 / 2 + 0.5 = 1
            npimg = img.numpy()                                     # transform Tensor to numpy array
            plt.imshow(np.transpose(npimg, (1, 2, 0)))              # array is: 3x32x32 but plt.imshow() needs WidthxHightxChannel (32x32x3); that's what transpose is doing
            plt.show()

        # get some random training images
        # dataiter = iter(trainloader)
        # images, labels = dataiter.next()                          # split the value of dataiter into two variables!

        for id, (images, labels) in enumerate(trainloader):
            # show images
            showpicture(images[id])
            # print labels to console
            print("{} ".format(classes[labels[0]]))
            break

