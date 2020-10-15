# unsing the Neural Network from NeuralNetworksTut für MNIST Classification

from __future__ import print_function
import torch

# Data Loading and normalizing using torchvision
import torchvision
import torchvision.transforms as transforms

## If running on Windows and you get a BrokenPipeError, try setting
# the num_worker of torch.utils.data.DataLoader() to 0.
transform = transforms.transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307, ), (0.3081, ))],
)

#transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=False, num_workers=0)

def showexample():
    # show images for fun
    import matplotlib.pyplot as plt
    import numpy as np

    def imshow(img):
        npimg = img.numpy()
        plt.imshow(npimg[0], cmap="gray")
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # print lables before show picture otherwise the programm will not contunue
    print("".join("%5s" % labels))
    # make a grid with utils!
    imshow(torchvision.utils.make_grid(images))


# use torch.nn for neural networks and torch.nn.functional for functions!
import torch.nn as nn
import torch.nn.functional as F

# lets define a network: (always as class!)
class Net(nn.Module):

    # always need the init with super!
    def __init__(self):
        super(Net, self).__init__()
        # kernel
        # 1 input image channel, 6 output channels, 3x3 square convolution (3 is the filter which typically is 3 or 5)
        self.conv1 = nn.Conv2d(1, 6, 3)
        # first of conv2 has to be last of conv1!
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6 , 120) # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max pooling over a (2, 2) window
        # print(x.size(), "this is the size!!!!!")
        # x = self.conv1(x)
        # print(x.size(), "this is the size!!!!!")
        # x = F.relu(x)
        # print(x.size(), "this is the size!!!!!")
        # x = F.max_pool2d(x, 2)
        # print(x.size(), "this is the size!!!!!")
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # print(x.size())
        # if the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# if torch.cuda.is_available():
#         device = torch.device("cuda")
#         net = Net().to(device=device)
# else:
#     net = Net()
#
# print(net)

# define a loss function and optimizer
# TODO: define a loss function and optimizer

# train the network
# TODO: train the network

# test the network on training data
# TODO: test the network on training data

# TODO: Train on GPU