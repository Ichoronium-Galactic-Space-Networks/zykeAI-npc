import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        # Initialize weight matrices
        self.W_Q = np.random.randn(d_model, d_model)
        self.W_K = np.random.randn(d_model, d_model)
        self.W_V = np.random.randn(d_model, d_model)
        self.W_O = np.random.randn(d_model, d_model)
    
    def split_heads(self, x, batch_size):
        return np.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        Q = np.dot(Q, self.W_Q)
        K = np.dot(K, self.W_K)
        V = np.dot(V, self.W_V)
        
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        scores = np.matmul(Q, K.transpose((0, 1, 3, 2))) / np.sqrt(self.depth)
        
        if mask is not None:
            scores += (mask * -1e9)
        
        attention_weights = softmax(scores)
        output = np.matmul(attention_weights, V)
        
        output = np.reshape(output, (batch_size, -1, self.d_model))
        output = np.dot(output, self.W_O)
        
        return output, attention_weights

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len
        
        # Initialize positional encoding matrix
        self.pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)
    
    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
    
    def forward(self, x):
        return np.dot(np.maximum(0, np.dot(x, self.W1)), self.W2)

class TransformerLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
    
    def forward(self, x, mask):
        attention_output, _ = self.multihead_attention.forward(x, x, x, mask)
        ff_output = self.feed_forward.forward(attention_output)
        return ff_output

class Transformer:
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_seq_length):
        self.num_layers = num_layers
        self.layers = [TransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
    
    def forward(self, x, mask):
        x = self.positional_encoding.forward(x)
        for layer in self.layers:
            x = layer.forward(x, mask)
        return x
