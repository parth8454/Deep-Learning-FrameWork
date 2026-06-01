import math
from engine.value import Value

def relu(x):
    
    assert isinstance(x,Value)

    out = Value(0 if x.data < 0 else x.data ,(x,), 'ReLU')

    def _backward():

        x.grad += (out.data > 0) * out.grad

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