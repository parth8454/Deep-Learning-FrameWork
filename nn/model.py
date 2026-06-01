import random
from engine.value import Value
import nn.functional as F

class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-0.1, 0.1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return F.leaky_relu(act, alpha=0.1) if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]
    # this is A list concatenation
    

class Layer:

    def __init__(self,nin,nout,nonlin=True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]
    
    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True 

    def __call__(self, x):
        if not isinstance(x, list):
            x = [x]
        out = F.dropout(x, self.p, self.training)
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return []

class MLP:
    def __init__(self, nin, nouts, dropout_rate=0.0):
        sz = [nin] + nouts
        self.layers = []
        
        for i in range(len(nouts)):
            self.layers.append(Layer(sz[i], sz[i+1], nonlin=(i != len(nouts)-1)))
            
            if i != len(nouts)-1 and dropout_rate > 0:
                self.layers.append(Dropout(p=dropout_rate))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
            
    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False