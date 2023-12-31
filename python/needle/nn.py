"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(init.kaiming_uniform(
            in_features, out_features, (in_features, out_features)), device=device, dtype=dtype)
        if bias:
            self.bias = Parameter(init.kaiming_uniform(
                out_features, 1, (out_features, 1)).transpose(), device=device, dtype=dtype)
        else:
            self.bias = Parameter(init.zeros(
                1, out_features), device=device, dtype=dtype)

    def forward(self, X: Tensor) -> Tensor:
        return X @ self.weight + self.bias.broadcast_to((X.shape[0], self.out_features))


class Flatten(Module):
    def forward(self, X):
        B, *D = X.shape
        return X.reshape((B, np.prod(D)))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        result = x
        for mod in self.modules:
            result = mod(result)
        return result


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        N = logits.shape[0]
        C = logits.shape[1]
        lhs = ops.logsumexp(logits, axes=(1,)).sum()
        rhs = (logits * init.one_hot(C, y,
               device=logits.device, dtype=logits.dtype)).sum()
        return (lhs - rhs) / N


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(dim), device=device, dtype=dtype)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        N, D = x.shape
        if self.training:
            mean = x.sum(axes=(0,)) / N
            self.running_mean.data = (
                (1-self.momentum) * self.running_mean + self.momentum*mean).data
            mean = mean.reshape((1, D)).broadcast_to((N, D))

            var = ((x - mean)**2).sum(axes=(0,)) / N
            self.running_var.data = (
                (1-self.momentum) * self.running_var + self.momentum*var).data
            var = var.reshape((1, D)).broadcast_to((N, D))
        else:
            mean = self.running_mean.reshape(
                (1, self.dim)).broadcast_to(x.shape)
            var = self.running_var.reshape((1, self.dim)).broadcast_to(x.shape)
        std_x = (x - mean) / (var + self.eps)**0.5
        w = self.weight.reshape((1, D)).broadcast_to(x.shape)
        b = self.bias.reshape((1, D)).broadcast_to(x.shape)
        return w * std_x + b


class BatchNorm2d(BatchNorm1d):
    '''
    handle data of shape NxCxHxW
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        N, C, H, W = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((N*H*W, C))
        y = super().forward(_x).reshape((N, H, W, C))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(dim), device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        N, _ = x.shape
        mean = (ops.summation(x, axes=(-1,)) /
                self.dim).reshape((N, 1)).broadcast_to(x.shape)
        var = (ops.summation((x - mean)**2, axes=(-1,)) /
               self.dim).reshape((N, 1)).broadcast_to(x.shape)
        std_x = (x - mean) / (var + self.eps)**0.5
        w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return w * std_x + b


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p)
            return x * mask / (1-self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        y = self.fn(x)
        return y + x


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        in_size = kernel_size * kernel_size * in_channels
        out_size = kernel_size * kernel_size * out_channels
        self.weight = Parameter(init.kaiming_uniform(in_size, out_size, (kernel_size, kernel_size,
                                in_channels, out_channels)), device=device, dtype=dtype, requires_grad=True)
        bias_bound = 1.0 / (in_channels * kernel_size**2)**0.5
        if bias:
            self.bias = Parameter(init.rand(out_channels, low=-bias_bound,
                                  high=bias_bound), device=device, dtype=dtype, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        assert H == W
        x = x.transpose((1, 3)).transpose((1, 2))  # (N, H, W, C)
        padding = self.kernel_size // 2
        result = ops.conv(x, self.weight, stride=self.stride, padding=padding)
        if self.bias is not None:
            result += self.bias.reshape((1, 1, 1, self.out_channels)
                                        ).broadcast_to(result.shape)
        return result.transpose((1, 3)).transpose((2, 3))


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        bound = np.sqrt(1/hidden_size)
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound,
                              high=bound, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound,
                              high=bound, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-bound,
                                               high=bound, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-bound,
                                               high=bound, device=device, dtype=dtype, requires_grad=True))
        if nonlinearity == "relu":
            self.nonlinearity = ReLU()
        else:
            self.nonlinearity = ops.tanh
            
    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor containing the next hidden state
            for each element in the batch.
        """
        bs, _ = X.shape
        ret = X @ self.W_ih
        if self.bias:
            ret += self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))
        if h is not None:
            ret += h @ self.W_hh
            if self.bias:
                ret += self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))
        return self.nonlinearity(ret)
