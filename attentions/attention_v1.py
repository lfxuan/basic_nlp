import math
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

class SelfAttV1(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.hidden_dim = hidden_dim
    
    def forward(self, X):
        # X: [B, N, H]
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)
        
        # K [B, N, H] --> [B, H, N]
        qk_res = torch.matmul(Q, K.transpose(-1, -2))
        # attention weight [B, N, N]
        attn_weight = torch.softmax(qk_res / math.sqrt(self.hidden_dim), dim=-1)
        # output [B, N, H]
        output = torch.matmul(attn_weight, V)
        
        return output
        

if __name__ == "__main__":
    X = torch.randn(2, 3, 4)
    attn = SelfAttV1(hidden_dim=4)
    output = attn(X)
    print(output.shape)