# second Tutorial - Autograd: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py

import torch
# create Tensor and set 'require_grad = True' to track the computation
# tensor 1x4 bedeuted: 1 Reihe, 4 Spalten (1 row 4 columns)
x = torch.ones(2, 2, requires_grad=True)
print(x)