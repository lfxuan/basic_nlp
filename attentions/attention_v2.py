import math
import torch
import torch.nn as nn


class SelfAttV2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 3)
        self.dim = dim
        
    def forward(self, X):
        QKV = self.proj(X)
        Q, K, V = torch.split(QKV, self.dim, dim=-1)
        print(Q.shape, K.shape, V.shape)
        
        qk_res = torch.matmul(Q, K.transpose(-2, -1))
        att_weight = torch.softmax(qk_res / math.sqrt(self.dim), dim=-1)
        output = torch.matmul(att_weight, V)
        
        return output
    

if __name__ == "__main__":
    X = torch.randn(2, 3, 4)
    attn = SelfAttV2(dim=4)
    output = attn(X)
    print(output.shape)