# DL-Library

This project is a deep learning library like PyTorch, which can be used to build deep neural networks and do automatic differentiation for backpropagation.

It contains common neural network modules, e.g. Linear, Convolution, ReLU, BatchNorm, LayerNorm, Residual, Dropout...

It contains a complete Tensor library, having common Tensor operations, e.g. Add, MatMul, Reshape, Permute, Broadcast, Compact...

The tensors have two kinds of backend. It's either on CPU supported by NumPy or on GPU supported by CUDA. This means the neural network can be trained on either device.