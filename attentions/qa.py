import torch

"""
exp(-inf) = 0
softmax(x_i) = exp(x_i) / sum(exp(x_j))
所以 -inf 位置的 exp 为 0，对分母无贡献，自身输出也为 0。
"""


attn_logits = [[1.2, 0.5, 0.0, 0.0]]  # 假设后两个是 padding，被设为 0
attn_weights = torch.softmax(torch.tensor([1.2, 0.5, 0.0, 0.0]), dim=-1)
# = [0.43, 0.21, 0.13, 0.13] → **padding 位置仍有注意力！**
print(attn_weights)
attn_weights = torch.softmax(
    torch.tensor([1.2, 0.5, float("-inf"), float("-inf")]), dim=-1
)
print(attn_weights)
