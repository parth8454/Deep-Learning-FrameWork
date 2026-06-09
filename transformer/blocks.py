import numpy as np
from engine.value import Value
from transformer.attention import ScaledDotProductAttention

class TransformerBlock:
    def __init__(self,d_model: int,d_k: int,d_ff:int):

        self.d_model = d_model

        self.d_k = d_k
        self.d_ff = d_ff

        self.attention = ScaledDotProductAttention(d_model,d_k)

        ran1 = np.sqrt(6.0/(d_model + d_ff))

        ran2 = np.sqrt(6.0/(d_ff + d_model))
        self.W1_data = np.random.uniform(-ran1,ran1,(d_model, d_ff))

        self.W2_data = np.random.uniform(-ran2,ran2,(d_ff, d_model))

        self.W1 = Value(self.W1_data)

        self.W2 = Value(self.W2_data)

        self.gam_1 = Value(np.ones((1,1,d_model)))

        self.beta_1 = Value(np.zeros((1,1,d_model)))
        self.gam_2 = Value(np.ones((1,1,d_model)))
        
        self.beta_2 = Value(np.zeros((1, 1, d_model)))

        self.norm1_in = None

        self.norm1_out = None

        self.ffn_in = None
        self.ffn_hidden = None
        self.norm2_out = None

    def forward(self,x:Value)->Value:
        attn_out = self.attention(x)
        self.norm1_in =x.data + attn_out.data

        mean1 = np.mean(self.norm1_in,axis=1,keepdims=True)

        var1 = np.var(self.norm1_in, axis=-1, keepdims=True)

        self.norm1_out=(self.norm1_in-mean1)/np.sqrt(var1 + 1e-5)
        x_norm1 =self.norm1_out * self.gam_1.data + self.beta_1.data

        self.ffn_in = x_norm1
        self.ffn_hidden = np.matmul(self.ffn_in, self.W1.data)

        self.ffn_hidden = np.maximum(0, self.ffn_hidden)
        ffn_out = np.matmul(self.ffn_hidden, self.W2.data)

        norm2_in = x_norm1 + ffn_out

        mean2 = np.mean(norm2_in, axis=-1, keepdims=True)

        var2 = np.var(norm2_in, axis=-1, keepdims=True)
        self.norm2_out = (norm2_in - mean2)/np.sqrt(var2 + 1e-5)

        x_norm2 = self.norm2_out * self.gam_2.data + self.beta_2.data

        out = Value(x_norm2)
        out._prev = {x, self.W1, self.W2, self.gam_1, self.beta_1, self.gam_2, self.beta_2}

        def _backward():
            d_out = out.grad
            self.beta_2.grad = np.sum(d_out, axis=(0,1),keepdims=True)
            self.gam_2.grad = np.sum(d_out*self.norm2_out,axis=(0,1),keepdims=True)

            d_norm2_out = d_out*self.gam_2.data
            N = d_out.shape[-1]
            d_norm2_in = (N * d_norm2_out - np.sum(d_norm2_out, axis=-1, keepdims=True) - 
                          self.norm2_out * np.sum(d_norm2_out * self.norm2_out, axis=-1, keepdims=True))/N
            
            self.W2.grad = np.sum(np.matmul(np.swapaxes(self.ffn_hidden, -1, -2), d_norm2_in), axis=0)
            d_ffn_hidden = np.matmul(d_norm2_in, np.swapaxes(self.W2_data, -1, -2))

            d_ffn_hidden[self.ffn_hidden <= 0] = 0
            self.W1.grad = np.sum(np.matmul(np.swapaxes(self.ffn_in, -1, -2), d_ffn_hidden), axis=0)
            d_ffn_in = np.matmul(d_ffn_hidden, np.swapaxes(self.W1_data, -1, -2))

            d_x_norm1 = d_ffn_in + d_norm2_in

            self.beta_1.grad = np.sum(d_x_norm1, axis=(0, 1), keepdims=True)
            self.gam_1.grad = np.sum(d_x_norm1 * self.norm1_out, axis=(0, 1), keepdims=True)
        
            d_norm1_out = d_x_norm1 * self.gam_1.data
            d_norm1_in = (N * d_norm1_out - np.sum(d_norm1_out, axis=-1, keepdims=True) - 
                          self.norm1_out * np.sum(d_norm1_out * self.norm1_out, axis=-1, keepdims=True))/N
            
            attn_out.grad = d_norm1_in
            self.attention._backward()

            dX = d_norm1_in + x.grad
            if x.grad is None:
                x.grad = dX
            else:
                x.grad += dX

        out._backward = _backward
        return out
    
    def __call__(self, x: Value)->Value:
        return  self.forward(x)

