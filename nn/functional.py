import math
import random
from engine.value import Value

def relu(x):
    
    assert isinstance(x,Value)

    out = Value(0 if x.data < 0 else x.data ,(x,), 'ReLU')

    def _backward():

        x.grad += (out.data > 0) * out.grad

    out._backward = _backward
    return out

def leaky_relu(x, alpha=0.1):
    assert isinstance(x, Value)
    
    val = x.data if x.data > 0 else alpha * x.data
    out = Value(val, (x,), 'Leaky_ReLU')

    def _backward():
        dx = 1.0 if x.data > 0 else alpha
        x.grad += dx * out.grad
        
    out._backward = _backward
    return out


def tanh(x):

    assert isinstance(x,Value)

    val = x.data
    t = (math.exp(2 * val) - 1) / (math.exp(2 * val) + 1)
    out = Value(t, (x,),'tanh')

    def _backward():
        x.grad += (1 - t**2) * out.grad

    out._backward = _backward
    return out


def dropout(x_list, p=0.5, training=True):
    if not training or p == 0.0:
        return x_list
        
    keep_prob = 1.0 - p
    
    mask = [1 if random.random() < keep_prob else 0 for _ in x_list]
    
    return [out * (m / keep_prob) for out, m in zip(x_list, mask)]