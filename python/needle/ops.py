"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        pow_value = power_scalar(a, self.scalar-1)
        return (out_grad * pow_value * self.scalar,)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        out_grad_b = -1 * divide(a, power_scalar(b, 2))
        return divide(out_grad, b), out_grad * out_grad_b


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        shape_len = len(a.shape)
        transposed_axes = list(range(shape_len))
        if self.axes is None:
            transposed_axes[-2] = shape_len-1
            transposed_axes[-1] = shape_len-2
        else:
            transposed_axes[self.axes[0]] = self.axes[1]
            transposed_axes[self.axes[1]] = self.axes[0]
        return a.permute(transposed_axes)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.reshape(self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        input_shape_count = len(input_shape) - 1
        broadcasted_axes = []
        for dim in range(len(out_grad.shape)-1, -1, -1):
            if input_shape_count < 0:
                broadcasted_axes.append(dim)
                continue
            if input_shape[input_shape_count] != out_grad.shape[dim]:
                broadcasted_axes.append(dim)
            input_shape_count -= 1
        return summation(out_grad, tuple(broadcasted_axes)).reshape(input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return a.sum()
        if isinstance(self.axes, int):
            return a.sum(self.axes)
        result = a
        axes = sorted(list(self.axes),reverse=True)
        for axis in axes:
            result = result.sum(axis)
        return result

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        grad_shape = list(input_shape)
        if self.axes is not None:
            if isinstance(self.axes, int):
                grad_shape[self.axes] = 1
            else:
                for axis in self.axes:
                    grad_shape[axis] = 1
        else:
            grad_shape = [1] * len(input_shape)
        grad = reshape(out_grad, grad_shape)
        return broadcast_to(grad, input_shape)

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a, grad_b = matmul(out_grad, transpose(b)), matmul(transpose(a), out_grad)
        if grad_a.shape != a.shape:
            grad_a = summation(grad_a, tuple(range(len(grad_a.shape)-2)))
        if grad_b.shape != b.shape:
            grad_b = summation(grad_b, tuple(range(len(grad_b.shape)-2)))
        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return a * (-1)

    def gradient(self, out_grad, node):
        return mul_scalar(out_grad, -1)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return divide(out_grad,node.inputs[0])


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * exp(a)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return (a > 0) * a

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        return out_grad * (a > 0)


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxz = Z.max(self.axes, keepdims=True)
        ret = array_api.log(array_api.exp(Z - maxz.broadcast_to(Z.shape)).sum(axis=self.axes, keepdims=True)) + maxz
        if self.axes is None:
            axes = list(range(len(Z.shape)))
        elif isinstance(self.axes, int):
            axes = [self.axes]
        else:
            axes = list(self.axes)
        
        if self.axes is not None:
            out_shape = [size for i, size in enumerate(Z.shape) if i not in axes]
        else:
            out_shape = [1]
        
        return ret.reshape(tuple(out_shape))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        if self.axes is not None:
            shape = [1] * len(Z.shape)
            if isinstance(self.axes, int):
                s = set([self.axes])
            else:
                s = set(self.axes)
            j = 0
            for i in range(len(shape)):
                if i not in s:
                    shape[i] = node.shape[j]
                    j += 1
            node_new = node.reshape(shape)
            grad_new = out_grad.reshape(shape)
        else:
            node_new = node
            grad_new = out_grad
        return grad_new.broadcast_to(Z.shape) * exp(Z - node_new.broadcast_to(Z.shape))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        return a.tanh()

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * (1- power_scalar(tanh(a),2))


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        new_shape = list(args[0].shape)
        new_shape.insert(self.axis, len(args))
        result = array_api.empty(new_shape, device=args[0].device)
        slices = []
        for idx in range(len(new_shape)):
            if idx == self.axis:
                slices.append(0)
            else:
                slices.append(slice(new_shape[idx]))
        for idx, arg in enumerate(args):
            slices[self.axis] = idx
            result[tuple(slices)] = arg
        return result
            
    def gradient(self, out_grad, node):
        return split(out_grad, axis=self.axis)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        old_shape = A.shape
        length = len(old_shape)
        num_tensors = old_shape[self.axis]
        tensors = []
        new_shape = [A.shape[idx] for idx in range(length) if idx != self.axis]
        slices = [slice(old_shape[idx]) if idx != self.axis else 0 for idx in range(length)]
        for idx in range(num_tensors):
            slices[self.axis] = idx
            tensors.append(A[tuple(slices)].reshape(new_shape))
        return tensors

    def gradient(self, out_grad, node):
        return stack(out_grad, axis=self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.flip(self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] += self.dilation * new_shape[axis]
        dilated_array = init.zeros(*new_shape, device=a.device, dtype=a.dtype, requires_grad=True)
        slices = []
        for idx in range(len(a.shape)):
            if idx in self.axes:
                slices.append(slice(0, new_shape[idx], self.dilation+1))
            else:
                slices.append(slice(0, new_shape[idx]))
        dilated_array.cached_data[tuple(slices)] = a
        return dilated_array.cached_data

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        shape = a.shape
        slices = []
        for idx in range(len(a.shape)):
            if idx in self.axes:
                slices.append(slice(0, shape[idx], self.dilation+1))
            else:
                slices.append(slice(0, shape[idx]))
        return a[tuple(slices)]

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        A = A.pad(((0,0),(self.padding, self.padding),(self.padding, self.padding),(0,0)))
        N,H,W,Cin = A.shape
        K,_,_,Cout = B.shape
        new_H, new_W = H-K+1,W-K+1
        Ns,Hs,Ws,Cs = A.strides
        inner_dim = K*K*Cin
        A = A.as_strided((N,new_H, new_W,K,K,Cin),(Ns,Hs,Ws,Hs,Ws,Cs)).reshape((N*new_H*new_W,inner_dim))
        out = A @ (B.reshape((inner_dim, Cout)))
        out = out.reshape((N,new_H,new_W,Cout))[:, ::self.stride, ::self.stride, :]
        return out

    def gradient(self, out_grad, node):
        X,Weights = node.inputs
        _,H,W,_ = X.shape
        K = Weights.shape[0]
        weights_flipped = flip(Weights, axes=(0,1)).transpose((2,3))
        dilated_out_grad=out_grad
        if self.stride > 1:
            dilated_out_grad = dilate(out_grad, axes=(1,2), dilation=self.stride-1)
        dX = conv(dilated_out_grad, weights_flipped, padding=K-1)
        dX = Tensor(dX.cached_data[:, self.padding:self.padding+H, self.padding:self.padding+W, :], dtype=dX.dtype, device=dX.device)
        X = transpose(X, (0,3)) # convert to shape (Cin,H,W,N)
        dilated_out_grad = dilated_out_grad.transpose((0,2)).transpose((0,1)) # convert to shape (Hhat,What,N,Cout)
        dW = conv(X, dilated_out_grad, padding=self.padding) # result is of shape (Cin,K,K,Cout)
        dW = dW.transpose((0,2)).transpose((0,1))
        return dX, dW


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



