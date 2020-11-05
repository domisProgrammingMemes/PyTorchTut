# Autoencoder first try from https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1

from __future__ import print_function

if __name__ == "__main__":
    import torch

    # Data Loading and normalizing using torchvision
    import torchvision
    import torchvision.transforms as transforms
    # use torch.nn for neural networks and torch.nn.functional for functions!
    import torch.nn as nn
    import torch.nn.functional as F
    # import torch optim for optimizer
    import torch.optim as optim

    # Path for Data
    data_path = './data'
    # Path to save and load model
    net_path = './AE_net.pth'

    # set up the divice (GPU or CPU) via input prompt
    cuda_true = input("Use GPU? (y) or (n)?")
    if cuda_true == "y":
        device = "cuda"
    else:
        device = "cpu"
    print("Device:", device)

    # Hyperparameters
    num_epochs = 1
    train_batch_size = 64
    test_batch_size = 64
    learning_rate = 0.001

    # imports to show pictures
    import matplotlib.pyplot as plt
    import numpy as np




