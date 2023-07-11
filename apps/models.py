import numpy as np
import math
import needle.nn as nn
import needle as ndl
import sys
sys.path.append('./python')
np.random.seed(0)


class ConvBN(ndl.nn.Module):
    def __init__(self, a, b, k, s, device=None, dtype="float32"):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model = nn.Sequential(nn.Conv(a, b, k, s, device=device, dtype=dtype), nn.BatchNorm2d(
            dim=b, device=device, dtype=dtype), nn.ReLU())

    def forward(self, x):
        return self.model(x)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        modules = [ConvBN(3, 16, 7, 4, device=device, dtype=dtype),
                   ConvBN(16, 32, 3, 2, device=device, dtype=dtype),
                   nn.Residual(nn.Sequential(ConvBN(32, 32, 3, 1, device=device,
                                             dtype=dtype), ConvBN(32, 32, 3, 1, device=device, dtype=dtype))),
                   ConvBN(32, 64, 3, 2, device=device, dtype=dtype),
                   ConvBN(64, 128, 3, 2, device=device, dtype=dtype),
                   nn.Residual(nn.Sequential(ConvBN(128, 128, 3, 1, device=device,
                                             dtype=dtype), ConvBN(128, 128, 3, 1, device=device, dtype=dtype))),
                   nn.ReLU(),
                   nn.Flatten(),
                   nn.Linear(128, 128, device=device, dtype=dtype),
                   nn.ReLU(),
                   nn.Linear(128, 10, device=device, dtype=dtype)]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset(
        "data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(
        cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)
