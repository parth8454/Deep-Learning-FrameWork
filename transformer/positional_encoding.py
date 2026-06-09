import numpy as np
from engine.value import Value

class PositionalEncoding:
    
    def __init__(self,max_seq_len : int, embedding_dim: int):

        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        pe = np.zeros((max_seq_len,embedding_dim))

        position = np.arrange(max_seq_len).reshape(-1,1)

        div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe[np.newaxis, ...]
    def forward(self,x: Value) -> Value:
        
        seq_len = x.data.shape[1]

        wave_slice = self.pe[:, :seq_len, :]

        out = x + wave_slice

        return out
    
    def __call__(self, x: Value )->Value:
            return self.forward(x)