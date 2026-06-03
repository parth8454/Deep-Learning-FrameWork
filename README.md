# Deep-Neural 🧠

Welcome everyone! If you have ever looked at massive deep learning frameworks like PyTorch or TensorFlow and wondered what is actually happening inside that black box, you are in the right place.

## What is in this repository?

**Deep-Neural** is a mathematically rigorous, scratch-built Deep Learning framework and Automatic Differentiation (Autograd) engine.

This entire framework was built from absolute zero using **pure Python**. There are no C++ backends. There is no NumPy doing the heavy matrix lifting. Every single mathematical operation, forward pass, and backpropagation step is handled by a custom-built dynamic computational graph.

Despite being written in pure Python, it features a modern, PyTorch-like API and implements state-of-the-art architectural standards, including:

- A **Reverse-Mode Autodiff Engine** (Directed Acyclic Graph)
- **Adaptive Moment Estimation** (Adam Optimizer)
- **Xavier/He Weight Initialization**
- **Inverted Dropout Regularization**
- **Non-Linear Activations** (ReLU, Leaky ReLU, Tanh)

It is capable of training multi-layer perceptrons on complex, high-dimensional datasets like **MNIST**.

---

## 📂 Folder Structure

Before diving into the math, here is a bird's-eye view of how the framework is organized:

deep-neural/
├── engine/
│   ├── __init__.py
│   └── value.py         # The core Autograd Scalar Engine
├── nn/
│   ├── __init__.py
│   ├── functional.py    # Decoupled mathematical & activation operations
│   ├── losses.py        # Error calculation criteria
│   ├── model.py         # Neural architecture (Neuron, Layer, Dropout, MLP)
│   └── optim.py         # Optimization algorithms (SGD, Adam)
└── tests/
    ├── test_engine.py       # Unit test for the backpropagation graph
    ├── train_digits.py      # Mini-MNIST (8x8) training script
    └── train_real_mnist.py  # Full MNIST (28x28) training script

---

## 🗺️ The Game Plan

Building a deep learning framework from scratch can seem completely overwhelming. To make this easy to digest, this documentation breaks down the repository step by step, exactly how it was built:

1. **Phase 1: The Engine** (`value.py`) — How we hijacked basic Python math to build a Memory Graph that remembers equations.
2. **Phase 2: The Math** (`functional.py` & `losses.py`) — How we added complex calculus like Activations and Error Calculators without clogging up the core engine.
3. **Phase 3: The Brain** (`model.py`) — How we stacked those math equations into physical Neurons, Layers, and Multi-Layer Perceptrons.
4. **Phase 4: The Ferrari Engine** (`optim.py`) — How we built the Adam Optimizer to automate the calculus and step the weights down the mountain.
5. **Phase 5: The Execution** (`tests/`) — How we actually use this framework to teach a pile of raw Python math how to read human handwriting.

---

## 🔬 Deep Dive: The Autograd Engine (`value.py`)

The `engine/value.py` file is the beating heart of the Deep-Neural framework. It contains exactly one class: the `Value` object.

This single file replaces thousands of lines of C++ code found in traditional frameworks like PyTorch by elegantly utilizing Python's object-oriented features to build a dynamic computation graph.

### 1. The `Value` Object: Memory and Gradients

A standard Python float (like `x = 5.0`) only holds a value. When it undergoes a mathematical operation, it forgets its origins. In deep learning, variables need to remember exactly how they were created so they can calculate their partial derivatives later.

To solve this, we wrap every number in a `Value` object.

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data                  # The actual scalar number
        self.grad = 0.0                   # The derivative of the output with respect to this value
        self._backward = lambda: None     # The function that calculates the local gradient
        self._prev = set(_children)       # The "parents" that created this value
        self._op = _op                    # The mathematical operation used to create this value
```

- **`data`**: The forward pass value.
- **`grad`**: Accumulates the gradient during the backward pass. Initializes at `0.0`.
- **`_prev`**: A set of the parent `Value` objects. This is what connects isolated numbers into a massive Directed Acyclic Graph (DAG).

---

### 2. Operator Overloading: Building the Graph

To make `Value` objects behave like normal numbers, we use Python's "magic methods" to intercept standard math operators. When you write `a + b`, Python calls `a.__add__(b)`.

Instead of just returning the sum, our custom `__add__` method creates a brand new `Value` object, links `a` and `b` as its parents, and defines exactly how to calculate the local derivative for addition

```python
def __add__(self, other):
    # Ensure 'other' is a Value object
    other = other if isinstance(other, Value) else Value(other)

    # 1. Forward Pass: Calculate the new data
    out = Value(self.data + other.data, (self, other), '+')

    # 2. Backward Pass Logic: Define the local Chain Rule
    def _backward():
        # The local derivative of addition is 1.0
        self.grad  += 1.0 * out.grad
        other.grad += 1.0 * out.grad

    out._backward = _backward
    return out
```

By overloading `__add__`, `__mul__`, `__pow__`, and others, every mathematical equation naturally strings itself together into a traceable memory graph without the user ever realizing it.

---

### 3. The Topological Sort

Calculus must be executed in a strict order. You cannot calculate the gradient of an input node until you have fully calculated the gradients of the nodes that depend on it.

To solve this, we use a **Topological Sort**. This algorithm starts at the final output node (the loss) and recursively traces backward through all the `_prev` parents, adding them to a list only after all their children have been processed.

```python
topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
build_topo(self)
```

> **Note on limits:** Because deep neural networks create massive, deeply nested graphs, this recursive function can hit Python's default recursion limit (1,000 steps). When training large models, you must increase the system recursion limit using `sys.setrecursionlimit()`.

---

### 4. The Backward Pass (Reverse-Mode Autodiff)

Once the topological graph is built, executing the backpropagation is incredibly simple.

We call `.backward()` on the final output node (usually the Loss). The engine sets the base gradient to `1.0` (because a variable's derivative with respect to itself is exactly 1). It then reverses the topologically sorted list and triggers the `_backward()` closure stored inside every single node.

```python
def backward(self):
    # Build the topological graph
    topo = []
    # ... (build_topo logic here) ...

    # Initialize the base gradient
    self.grad = 1.0

    # Traverse the graph backward and apply the Chain Rule
    for node in reversed(topo):
        node._backward()
```

As the loop iterates, the Chain Rule ripples backward perfectly, accumulating partial derivatives into the `.grad` attribute of every single parameter in the network, leaving them perfectly primed for the Optimizer to step them.

---

## ➗ Deep Dive: The Math (`functional.py` & `losses.py`)

While `value.py` handles the fundamental calculus and graph tracking, cramming complex neural network mathematics into the core engine would violate the principle of **Separation of Concerns**.

To solve this, Deep-Neural mirrors PyTorch's architecture by creating a standalone functional library (`nn/functional.py`) and a dedicated module for error calculation (`nn/losses.py`). These files act as a bridge, utilizing the basic `Value` objects to execute complex, multi-step mathematical operations while automatically preserving the Autograd graph.

---

### 1. The Functional API (`nn/functional.py`)

The functional module contains pure mathematical operations. These functions take `Value` objects (or lists of them) as inputs, apply mathematical transformations, and return new `Value` objects with carefully defined local derivatives.

#### Non-Linear Activations

If a neural network only uses standard multiplication and addition, stacking 100 layers is mathematically identical to a single layer. Activation functions introduce **non-linearity**, allowing the network to "bend" its decision boundaries to learn complex patterns.

**ReLU (Rectified Linear Unit):** The standard activation. It acts as a gate — if the input is positive, it passes through unchanged; if it is negative, it flatlines to 0.

> **Derivative:** `1` if `x > 0`, else `0`.

**Leaky ReLU:** Standard ReLU has a fatal flaw known as the **"Dying ReLU" problem**. If a neuron's weights shift too far into the negative, it outputs `0` forever, killing its gradient. Leaky ReLU fixes this by allowing a tiny fractional signal (`alpha`) to pass through when negative.

```python
def leaky_relu(x, alpha=0.1):
    val = x.data if x.data > 0 else alpha * x.data
    out = Value(val, (x,), 'Leaky_ReLU')

    def _backward():
        # Derivative is 1 if positive, alpha if negative
        dx = 1.0 if x.data > 0 else alpha
        x.grad += dx * out.grad

    out._backward = _backward
    return out
```

**Tanh (Hyperbolic Tangent):** Squishes any real-valued number into a smooth, bounded curve between `-1.0` and `1.0`. Used frequently in recurrent architectures.

#### Inverted Dropout Regularization

To prevent the network from memorizing the training data (**overfitting**), Dropout randomly "blinds" a percentage of the network's neurons during training. This forces the remaining neurons to learn robust, generalized features.

Deep-Neural implements **Inverted Dropout**. If we drop 50% of our neurons, the total sum of the layer's output is cut in half. Instead of scaling the weights down during validation to compensate (which is slow and complex), Inverted Dropout scales the surviving neurons **up** during training by a factor of $\frac{1}{1 - p}$.

```python
def dropout(x_list, p=0.5, training=True):
    if not training or p == 0.0:
        return x_list  # Do nothing during validation!

    keep_prob = 1.0 - p
    mask = [1 if random.random() < keep_prob else 0 for _ in x_list]

    # Scale up the survivors so the expected value remains constant
    return [out * (m / keep_prob) for out, m in zip(x_list, mask)]
```

Because the inputs are `Value` objects, the Autograd engine automatically handles the calculus for this multiplication. Dead neurons get a gradient of `0`, and surviving neurons get perfectly scaled gradients!

---

### 2. The Criterion (`nn/losses.py`)

The Autograd engine needs a starting point — a single, final scalar number that represents exactly how "wrong" the network's current predictions are. This is the job of the **Loss Criterion**.

#### MSELoss (Mean Squared Error)

Deep-Neural implements `MSELoss`, which calculates the squared distance between the network's predicted logits and the true labels (usually one-hot encoded).

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}} - y_{\text{pred}})^2$$

```python
class MSELoss:
    def __call__(self, y_true, y_pred):
        # Calculate the squared differences
        differences = [(yt - yp)**2 for yt, yp in zip(y_true, y_pred)]

        # Sum them up and divide by the total number of elements
        total_loss = sum(differences)
        return total_loss * (1.0 / len(y_true))
```

#### The Magic of the Graph

Look closely at the `MSELoss` code — there is no `_backward()` closure defined here!

Because `y_pred` is a list of `Value` objects, operations like subtraction (`-`), exponentiation (`**2`), and addition (`sum`) trigger the overloaded magic methods inside `value.py`. The `MSELoss` class simply strings these basic operations together.

When it returns `total_loss`, it is returning the final `Value` node at the very end of a massive computation graph. Calling `total_loss.backward()` triggers the topological sort that ripples all the way back through the loss calculation, through the functional activations, and into the network's weights.

---

## 🧠 Deep Dive: The Brain (`model.py`)

While `value.py` handles the raw calculus and `functional.py` provides the complex math, a neural network needs a **physical structure** to organize these operations.

The `nn/model.py` file builds the biologically-inspired architecture. It follows a strict Object-Oriented design, creating modular "Lego bricks" that can be stacked to build networks of any size. Every module exposes a `.parameters()` method that returns a list of its trainable `Value` objects, making it easy for the Optimizer to find and update them.

---

### 1. The Neuron: The Foundational Node

The `Neuron` represents a single node in the network. It takes an array of inputs, multiplies them by its own internal weights, adds a bias, and passes the result through a non-linear activation function.

#### The Weight Initialization Trap (Mini-Xavier)

A common pitfall in custom frameworks is initializing weights with completely random numbers (e.g., between `-1.0` and `1.0`). In deep networks, summing dozens of large random numbers causes the output to **explode**. The MSE loss then becomes incredibly high, causing the optimizer to panic and crush all weights to zero, effectively killing the network on Epoch 1.

To solve this, Deep-Neural implements a scaled-down version of **Xavier/He Initialization**. Weights are initialized using a very tight uniform distribution (between `-0.1` and `0.1`), keeping the initial mathematical variance stable and giving the optimizer room to learn safely.

```python
class Neuron:
    def __init__(self, nin, nonlin=True):
        # Mini-Xavier Initialization: Small, stable starting weights
        self.w = [Value(random.uniform(-0.1, 0.1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        # Calculate (w * x) + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)

        # Apply Leaky ReLU if this neuron is allowed to activate
        return F.leaky_relu(act, alpha=0.1) if self.nonlin else act
```

---

### 2. The Layer: Parallel Processing

A single `Neuron` can only look at one feature at a time. A `Layer` is simply a collection of Neurons that process the exact same input data **in parallel**, allowing the network to extract multiple different patterns simultaneously.

```python
class Layer:
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
```

Notice how clean this is — there is no complex math here. The `Layer` simply delegates the work to its `Neuron`s and collects the results.

---

### 3. The Dropout Module: The Regularization Brick

In earlier iterations of Deep-Neural, Dropout logic was hardcoded directly into the `Layer` class. This violated software engineering principles and made the code rigid.

Dropout was extracted into its own standalone module. It has no weights and no bias — it simply takes an input, passes it to the `functional.dropout` math, and returns the result. It acts as a strict "filter" block that can be placed anywhere in the network.

```python
class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True  # Default to training mode

    def __call__(self, x):
        # Passes the data to the functional math API
        out = F.dropout(x, self.p, self.training)
        return out[0] if len(out) == 1 else out
```

---

### 4. The MLP: The Master Container

The **Multi-Layer Perceptron (MLP)** is the master wrapper that assembles the final network. You pass it the number of inputs and a list of layer sizes, and it automatically stacks `Layer` and `Dropout` blocks in perfect sequence.

#### The Dead Output Layer Phenomenon

A critical architectural detail of the MLP is how it handles the **final layer**. If the output layer is passed through a ReLU activation, it can never output a negative number — restricting the network from producing true scalar logits and completely breaking accuracy calculation.

The MLP contains explicit logic to ensure the final layer never receives an activation function and never receives Dropout.

```python
class MLP:
    def __init__(self, nin, nouts, dropout_rate=0.0):
        sz = [nin] + nouts
        self.layers = []

        for i in range(len(nouts)):
            # Add the Layer. If it is the LAST layer, nonlin is set to False!
            is_last_layer = (i == len(nouts) - 1)
            self.layers.append(Layer(sz[i], sz[i+1], nonlin=not is_last_layer))

            # Add Dropout immediately after the layer (except the final layer)
            if not is_last_layer and dropout_rate > 0:
                self.layers.append(Dropout(p=dropout_rate))
```

#### State Management: Train vs. Eval

Neural networks must behave differently during **training** (where Dropout regularization is active) and **validation** (where 100% of the network's capacity is used for prediction).

The MLP exposes PyTorch-style mode switches — calling `nn.train()` sets all `Dropout` blocks to active, while `nn.eval()` safely bypasses them.

---

## 🏎️ Deep Dive: The Optimizers (`optim.py`)

Calculating the gradients using the Autograd engine only tells the network which **direction** it needs to move to reduce the error. The Optimizer is the engine that actually **physically moves the weights**.

The `nn/optim.py` file takes the list of `.parameters()` generated by your network and updates their `.data` values using the calculated `.grad` values.

---

### 1. SGD (Stochastic Gradient Descent)

SGD is the baseline optimization algorithm. It simply takes the gradient, multiplies it by a small **Learning Rate** ($\eta$), and subtracts it from the weight.

$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla J(\theta)$$

```python
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        # Gradients accumulate by default. We must reset them to 0 before every epoch!
        for p in self.parameters:
            p.grad = 0.0
```

While SGD works, it has a major flaw: it takes **blind, equal-sized steps**. If the mathematical landscape is highly curved, SGD will bounce back and forth wildly. If the landscape is flat, it will slow to a crawl.

---

### 2. Adam (Adaptive Moment Estimation)

Adam is the "Ferrari" of modern deep learning. Instead of taking blind steps, Adam calculates a **dynamic, custom learning rate for every single parameter individually** by tracking two things:

- **Momentum (First Moment, $m$):** A moving average of past gradients. If the gradient keeps pointing in the same direction, Adam builds up speed and accelerates down the hill.
- **Variance (Second Moment, $v$):** A moving average of past squared gradients. If a parameter's gradient is bouncing wildly, Adam acts as a shock absorber and shrinks its learning rate to stabilize it.

Because these moving averages start at `0.0`, they are biased towards zero at the beginning of training. Adam implements **Bias Correction** ($\hat{m}$ and $\hat{v}$) to forcefully scale them up during the first few epochs.

```python
class Adam:
    def __init__(self, parameters, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # Time step tracker

        # Initialize moment arrays for every single parameter
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            # 1. Update Momentum (m) and Variance (v)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)

            # 2. Bias Correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # 3. The Final Weight Update
            p.data -= self.lr * m_hat / (v_hat**0.5 + self.eps)
```

By implementing Adam in pure Python, Deep-Neural can train networks exponentially faster and avoid getting stuck in mathematical "saddle points."

---

## 🧪 Deep Dive: The Execution (`tests/`)

A mathematical framework is useless if it cannot be proven to work. The `tests/` directory contains the execution scripts that put the Autograd engine, the model architecture, and the optimizers together end-to-end.

---

### 1. Unit Testing the Graph (`test_engine.py`)

Before training a network, you must prove the calculus works. This script constructs a small equation like $y = (x \cdot 2) + z$, triggers `.backward()`, and compares the resulting `.grad` values against the known, mathematically proven partial derivatives.

If this file passes, the foundation of the framework is **mathematically sound**.

---

### 2. Rapid Prototyping (`train_digits.py`)

This is the daily-driver test script. It uses the Scikit-Learn `load_digits()` dataset — 1,797 images of hand-drawn digits scaled down to $8 \times 8$ pixels (64 total inputs).

**Architecture:** 64 Inputs → 16 Hidden Neurons → 10 Output Neurons

Because the graph is relatively small (~1,200 parameters), it runs incredibly fast on a standard CPU. The script implements an **80/20 Train-Validation split**, making it a rapid proving ground for testing new features like Leaky ReLU or Dropout before deploying them at scale.

---

## 🏁 Final Words & Next Steps

Building Deep-Neural was born out of a desire to completely escape **"tutorial hell."** You cannot truly understand how Artificial Intelligence works just by importing PyTorch and calling `.train()`. You have to build the math, break the limits, and assemble the graph yourself.

This framework is not meant to replace enterprise libraries. It is meant to be a **crystalline, readable, and mathematically pure educational tool**. If you are a student, a developer, or just someone fascinated by Deep Learning, I highly encourage you to clone this repository.

> Break the code. Change the activation functions. Write a new optimizer. Watch how the math reacts.

---

## 🤝 Contributing

If you find a bug in the calculus, want to implement a new optimizer like RMSprop, or just want to optimize the topological sort, feel free to open a **Pull Request**. Let's make this the best scratch-built learning framework on GitHub.

If this repository helped you finally understand what is happening inside the black box of Deep Learning, consider leaving a ⭐ **Star** on the repo!

---

*Keep building,*

**Paxton**