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
# cuda_true = input("Use GPU? (y) or (n)?")
# if cuda_true == "y":
#     device = "cuda"
# else:
#     device = "cpu"
# print("Device:", device)

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

# save a networks parameters for future use
def save_network(net: nn.Module, path):
    # save the network?
    save = input("Save net? (y) or (n)?")
    if save == "y":
        torch.save(net.state_dict(), path)
    else:
        pass

# load an existing network's parameters and safe them into the just created net
def load_network(net: nn.Module, path):
    # save the network?
    save = input("Load Network? (y) or (n)?")
    if save == "y":
        net.load_state_dict(net.state_dict(), net_path)
    else:
        pass

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

    # step 1: Load and normalizing the CIFAR10 training and test datasets using torchvision
    # the output of torchvision datasets are PILImages of range [0, 1].
    # transform to Tensors of normalized range [-1, 1]

    trainset = datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
    trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=0)

    testset = datasets.CIFAR10(root=data_path, train=False, transform=transform, download=True)
    testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=0)

    # classes from dataset (this is a tuple:https://www.w3schools.com/python/python_tuples.asp):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # show an example picture
    def example(dataloader):
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

        for id, (images, labels) in enumerate(dataloader):
            # show 1 image
            # showpicture(images[id])
            # print 1 labels to console
            # print("{} ".format(classes[labels[0]]))
            # with showpicture(torchvision.utils.make_grid(images)) torchvision will make 1 image out of batch_size images
            showpicture(torchvision.utils.make_grid(images))
            print(' '.join('%5s' % classes[labels[j]] for j in range(dataloader.batch_size)))

    # run showing an example out of traindata
    example(trainloader)

    # ----------------------------------------------------------------------------------------------------

    # step 2: Define a Convolutional Neural Network
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=5)
            self.conv2 = nn.Conv2d(8, 12, kernel_size=5)
            self.pool = nn.MaxPool2d(2)
            # for FC you first need to know what dimension the data will have --> result: 300
            self.fc1 = nn.Linear(300, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)


        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))                       # relu is typically used with conv! after conv: maxpool to reduce dimension
            x = self.pool(F.relu(self.conv2(x)))                       # relu is typically used with conv! after conv: maxpool to reduce dimension
            x = x.view(-1, self.num_flat_features(x))                  # view reshapes the tensor while keeping it's data -> changes x to a vector with 300 elements; each picture is a [1, 300]
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


        def num_flat_features(self, x):                                 # multiplies all values from the tensor to get number of features (e.g. [batch_size, 12x5x5] -> 300 (12x5x5)
            size = x.size()[1:]                                         # ignore batch size -> [1:] from 1 until end of Tensor
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    # create an instance of the net -> CNN
    mynet = CNN()
    # load_network(mynet, net_path)

    # ----------------------------------------------------------------------------------------------------

    # step 3: Define a loss function
    # I'll use CrossEntropyLoss as criterion and SGD for the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mynet.parameters(), lr=learning_rate, momentum=momentum)

    # ----------------------------------------------------------------------------------------------------

    # step 4: Train the network on the training data
    for epoch in range(num_epochs):                                     # loop over dataset (traindata) multiple times #num_epochs
        running_loss = 0.0
        for batch_idx, data in enumerate(trainloader, 0):
            # get the inputs, data is a list of [inputs, labels] and also batch id
            inputs, labels = data

            # zero the gradients
            optimizer.zero_grad()

            # forward
            outputs = mynet(inputs)
            loss = criterion(outputs, labels)
            # + backward
            loss.backward()
            # + optimize
            optimizer.step()

            # print some statistics
            running_loss += loss.item()
            if batch_idx % 50 == 49:                                # print every X (=50) batches
                print("[Epoch: {}, Batch: {} - loss: {:.4f}".format(
                    epoch + 1, batch_idx + 1, running_loss / 50)
                )
                running_loss = 0.0

    # save_network(mynet, net_path)
    print("Training finished")

    # ----------------------------------------------------------------------------------------------------

    # TODO: step 5: Test the network on the test data
    example(testloader)


    # mynet = CNN()
    # for id, (images, labels) in enumerate(trainloader):
    #     outputs = mynet(images)
    #     if id == 0:
    #         break

    # TODO: GPU
