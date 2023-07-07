"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            if not param.requires_grad or param.grad is None:
                continue
            new_u = self.momentum * self.u.get(param, 0) + (1-self.momentum) * (
                param.grad.data + self.weight_decay * param.data)
            self.u[param] = new_u
            new_param = param.data - self.lr * new_u
            param.data = ndl.Tensor(
                new_param.data, device=param.device, dtype=param.dtype, requires_grad=False).data

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params:
            if not param.requires_grad or param.grad is None:
                continue
            grad_wd = param.grad.data + self.weight_decay * param.data
            new_m = self.beta1 * \
                self.m.get(param, 0) + (1-self.beta1) * grad_wd
            new_v = self.beta2 * \
                self.v.get(param, 0) + (1-self.beta2) * grad_wd**2
            self.m[param] = new_m
            self.v[param] = new_v
            new_m /= 1-self.beta1**self.t
            new_v /= 1-self.beta2**self.t
            new_param = param.data - self.lr * new_m / (new_v**0.5 + self.eps)
            param.data = ndl.Tensor(
                new_param.data, device=param.device, dtype=param.dtype, requires_grad=False).data

