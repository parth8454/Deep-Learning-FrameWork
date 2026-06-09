import numpy as np
from engine.value import Value

class ScaledDotProductAttention:
    def __init__(self,d_model: int, d_k: int):

        self.d_model = d_model
        self.d_k = d_k
        
        ran = np.sqrt(6/(d_k + d_model))

        w_q_data = np.random.uniform(-ran,ran,(d_model,d_k))
        w_k_data = np.random.uniform(-ran,ran,(d_model,d_k))
        w_v_data = np.random.uniform(-ran,ran,(d_model,d_k))

        self.W_Q = Value(w_q_data)
        self.W_K = Value(w_k_data)
        self.W_V = Value(w_v_data)

    def forward(self,x: Value)->Value:

        self.x_data = x.data

        self.Q = np.matmul(self.x_data, self.W_Q.data)
        self.K = np.matmul(self.x_data, self.W_K.data)
        self.V = np.matmul(self.x_data, self.W_V.data)

        K_T = np.swapaxes(self.K,-1,-2)
        scores = np.matmul(self.Q, K_T)

        scaled_scores = scores / np.sqrt(self.d_k)

        exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
        self.attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        output_data = np.matmul(self.attention_weights, self.V)

        out = Value(output_data)
        out._prev = {x, self.W_Q, self.W_K, self.W_V}

        def _backward():
            d_out = out.grad
            d_V = np.matmul(np.swapaxes(self.attention_weights, -1, -2), d_out)
            d_attention_weights = np.matmul(d_out, np.swapaxes(self.V, -1, -2))
            d_softmax = self.attention_weights * (d_attention_weights - np.sum(d_attention_weights * self.attention_weights, axis=-1, keepdims=True))
            
            d_scaled = d_softmax/np.sqrt(self.d_k)

            d_Q = np.matmul(d_scaled, self.K)
            d_K = np.matmul(np.swapaxes(d_scaled, -1, -2), self.Q)

            dW_Q = np.sum(np.matmul(np.swapaxes(self.x_data, -1, -2), d_Q), axis=0)

            dW_K = np.sum(np.matmul(np.swapaxes(self.x_data, -1, -2), d_K), axis=0)
            dW_V = np.sum(np.matmul(np.swapaxes(self.x_data, -1, -2), d_V), axis=0)

            for weight_node, grad_array in [(self.W_Q, dW_Q), (self.W_K, dW_K), (self.W_V, dW_V)]:
                if weight_node.grad is None:
                    weight_node.grad = grad_array
                else:
                    weight_node.grad += grad_array
            
            dX = (np.matmul(d_Q, np.swapaxes(self.W_Q_data,-1,-2))+
                   np.matmul(d_K, np.swapaxes(self.W_K_data,-1,-2))+
                   np.matmul(d_V, np.swapaxes(self.W_V_data,-1,-2)))
            
            if x.grad is None:
                x.grad = dX
            else:
                x.grad += dX

            out._backward = _backward

            return out
        
        def __call__(self, x: Value) -> Value:
            return self.forward(x)