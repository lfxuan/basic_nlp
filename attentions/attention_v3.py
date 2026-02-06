"""
attention 计算的时候有 dropout，而且是比较奇怪的位置
attention 计算的时候一般会加入 attention_mask，因为样本会进行一些 padding 操作；
MultiHeadAttention 过程中，除了 QKV 三个矩阵之外，还有一个 output 对应的投影矩阵，因此虽然面试让你写 SingleHeadAttention，但是依然要问清楚，是否要第四个矩阵？
"""

import torch
import torch.nn as nn
import math

class SelfAttV3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        self.output_proj = nn.Linear(dim, dim)
        
    def forward(self,X, mask=None):
        # X [B, N, D]
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)
        
        # att_weight [B, N, N]
        qk_res = torch.matmul(Q, K.transpose(-2, -1))
        att_weight = qk_res / math.sqrt(self.dim)
        if mask is not None:
            # mask需要将无效设为很大的负数，确保softmax 的结果无效位置为0
            att_weight = att_weight.masked_fill(mask==0, float("-1e20"))  
        # print(att_weight)
        att_weight = torch.softmax(att_weight, dim=-1)
        # print(att_weight)
        
        # res [B, N, D]
        # 为了正则化注意力机制，防止模型过度依赖少数 key，提升泛化能力，Transformer 原论文 + PyTorch/TensorFlow 官方实现均采用
        att_weight = self.dropout(att_weight)
        output = torch.matmul(att_weight, V)
        res = self.output_proj(output)
        
        return res


if __name__ == "__main__":
    X = torch.randn(3, 4, 2)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
    print(mask.shape)
    # print(mask.unsqueeze(1).shape)
    # mask = mask.unsqueeze(1).repeat(1, X.shape[1], 1)  # 占内存
    mask = mask.unsqueeze(1).expand(-1, X.shape[1], -1)
    print(mask)
    print(mask.shape)
    att = SelfAttV3(dim=2)
    res = att(X, mask)
    print(res.shape)
    
