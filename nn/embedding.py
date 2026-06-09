# well i love my laptop and i really dont want to see him suffer hence,
# i am using numpy for the heavy lifiting here

import numpy as np
from engine.value import Value
# our needs to handle 3 diff things 1)Initialization 2)forward_pass 3)Backward_pass

class Embeddings:

    def __init__(self,vocab_size: int,embedding_dim: int):
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        ran = np.sqrt(6/(vocab_size + embedding_dim))

        weight_data = np.random.uniform(-ran,ran,(vocab_size,embedding_dim))

        self.weights = Value(weight_data)

        self.last_input = None

    def forward(self,idx: np.ndarray) -> Value:

        self.last_input = idx

        output_data = self.weights.data[idx]

        out = Value(output_data)

        out._prev = {self.weights}    

        def _backward():
            
            grad_weights = np.zeros_like(self.weights.data)

            np.add.at(grad_weights, self.last_input, out.grad)

            if self.weights.grad is None:
                self.weights.grad = grad_weights
            else:
                self.weights.grad += grad_weights

        out._backward = _backward
        return out  