# DL-Library

This project is a deep learning library like PyTorch, which can be used to build deep neural networks and do automatic differentiation for backpropagation.

It contains common neural network modules, e.g. Linear, Convolution, ReLU, BatchNorm, LayerNorm, Residual, Dropout...

It contains a complete Tensor library, having common Tensor operations, e.g. Add, MatMul, Reshape, Permute, Broadcast, Compact...

The tensors support two kinds of backend devices, CPU and CUDA.

I have trained a 9-layer ResNet for classification on CIFAR-10, which can achieve 80+% accuracy in 25 epoch.
