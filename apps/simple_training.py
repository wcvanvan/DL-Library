import time
from models import *
from needle import backend_ndarray as nd
import needle.nn as nn
import needle as ndl
import sys
sys.path.append('../python')

device = ndl.cpu()

### CIFAR-10 training ###


def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """

    np.random.seed(4)
    correct, loss_total, batch_cnt, total_cnt = 0., 0., 0, 0
    if opt:
        model.train()
    else:
        model.eval()
    for X, y in dataloader:
        N = X.shape[0]
        total_cnt += N
        batch_cnt += 1
        if opt:
            opt.reset_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        correct += (pred.numpy().argmax(axis=1) == y.numpy()).sum()
        if opt:
            loss.backward()
            opt.step()
        loss_total += loss.numpy()

    return correct / total_cnt, loss_total / batch_cnt


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
                  lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for idx in range(n_epochs):
        acc, loss = epoch_general_cifar10(dataloader, model, loss_fn=loss_fn(), opt=opt)
        print(f"epoch {idx}; accuracy {acc}; loss {loss}")
    return acc, loss


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    return epoch_general_cifar10(dataloader, model, loss_fn(), opt=None)



if __name__ == "__main__":
    device = ndl.cuda()
    dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    dataloader = ndl.data.DataLoader(\
            dataset=dataset,
            batch_size=128,
            shuffle=True,
            device=device,
            dtype="float32")
    model = ResNet9(device=device, dtype="float32")
    train_cifar10(model, dataloader, n_epochs=20, optimizer=ndl.optim.Adam,
        lr=0.001, weight_decay=0.001)
    dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=False)
    evaluate_cifar10(model, dataloader)
