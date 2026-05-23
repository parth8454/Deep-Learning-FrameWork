# Autograd Engine & Neural Network Framework

This project is a lightweight, fully functional automatic differentiation engine and deep learning framework built entirely from scratch in Python. It features a custom scalar-valued autograd engine that dynamically builds a Directed Acyclic Graph (DAG) to automatically calculate gradients via backpropagation. On top of this mathematical foundation, the framework provides a PyTorch-style neural network API—complete with neurons, layers, multi-layer perceptrons (MLPs), loss functions, and an SGD optimizer—designed to explicitly demystify the "black box" of modern machine learning.

## Future Project Structure

```text
deep-neural/
├── engine/
│   ├── __init__.py
│   └── value.py               # Core autograd engine and DAG topological sort
├── nn/
│   ├── __init__.py
│   ├── model.py               # Neuron, Layer, and MLP architectures
│   ├── optim.py               # Stochastic Gradient Descent (SGD) optimizer
│   └── losses.py              # Extracted loss functions (MSE, etc.)
├── tests/
│   └── test_engine.py         # Unit tests to verify gradient accuracy
├── examples/
│   ├── 01_autograd_test.ipynb # Sandbox for engine calculus
│   └── 02_training_loop.ipynb # End-to-end model training 
├── .gitignore
└── README.md