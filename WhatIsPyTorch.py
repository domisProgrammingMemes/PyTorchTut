# this will be the master-branch Tutorial
import torch
print(torch.cuda.is_available())

# youtube tutorial: https://www.youtube.com/playlist?list=PLNmsVeXQZj7rx55Mai21reZtd_8m-qe27
# Test comment

# first Tutorial - What is PyTorch
from __future__ import print_function
# tensor 1x4 bedeuted: 1 Reihe, 4 Spalten (1 row 4 columns)


def partOne():
    x = torch.empty(5, 3)
    print(x)

    y = torch.rand(5, 3)
    print(y)

    z = torch.zeros(5, 3, dtype=torch.long)
    print(z)

    a = torch.tensor([5.5, 3])
    print(a)

    # or create tensor based on existing tensor
    a = a.new_ones(5, 3, dtype=torch.double)  # new_* take in sizes
    print(a)

    # override a again
    a = torch.rand_like(a, dtype=torch.float)
    print(a)

    print(a.size())

    b = torch.tensor([[2, 2, 2],
                      [2, 2, 2],
                      [2, 2, 2],
                      [2, 2, 2],
                      [2, 2, 2]], dtype=torch.float)
    print(b)
    print(b.dtype)

    # addition in 2 ways
    # 1 syntax
    # print(a + b)

    # 2 syntax
    # print(torch.add(a, b))

    # providing output tensor as argument
    result = torch.empty(b.size())  # size will be expanded if needed?
    print("result while empty: " + "\n" + str(result))
    torch.add(a, b, out=result)  # out specifies where the result should be written to
    print("result after addition of a and b\n" + str(result))

    # in-place addition
    # note: every action with an _ changes the tensor!

    b.add_(a)
    print(b)  # b is no longer 2s!

    # Numpy-like indexing
    print(b[:, 1])  # print second column
    print(b[0, :])  # print first row

def partTwo():
    # Resizing/reshaping a tensor using torch.view
    c = torch.randn(4, 4)
    d = c.view(16)  # reshape to a 1 dimensional tensor with 16x1
    e = c.view(-1,
               8)  # reshape to 2 dimensions with 8 x 2; the size of -1 is inferred (gefolgert) from the other dimension
    f = c.view(1, 16)  # difference between [16] and [1, 16] ?

    print(c, c.size())
    print(d, d.size())
    print(e, e.size())
    print(f, f.size())

    # get python value
    x = torch.randn(1)
    print(x)
    print(x.item())


def partThree():
    # numpy bridge
    a = torch.ones(5)
    print("this is a: " + str(a))

    b = a.numpy()
    print("this is b: " + str(b))

    a.add_(1)
    print("this is a after the addition of 1 in each column: " + str(a))
    print("this is b after the addition of 1 in the a tensor, but b is tied to a by memory: " + str(b))

    # converting numpy array to Torch Tensor
    import numpy as np
    a = np.ones(5)
    print("this is a befor calc: " + str(a))
    b = torch.from_numpy(a)
    np.add(a, 1, out=a)
    print("this is a after calc: " + str(a))
    print("this is b: " + str(b))


def partFour():
    # CUDA Tensors
    # tensors can be moved onto any device unsing .to method

    # let us run this cell only if CUDA is available
    # use torch.device objects to move tensors in and out of GPU
    x = torch.randn(1)
    print("x in the beginning: " + str(x))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        y = torch.ones_like(x, device=device)
        x = x.to(device)
        z = x + y
        print("this is z: " + str(z))
        print("z to cpu and change dtype: " + str(z.to("cpu", torch.double)))


partFour()

# Tutorial - What is PyTorch end