import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLoRALayer(nn.Module):
    def __init__(
        self, in_features, out_features, merge=False, rank=8, lora_alpha=16, dropout=0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.rank = rank
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        if rank > 0:
            self.lora_a = nn.Parameter(torch.zeros(out_features, rank))
            # lora_a 初始化参数为高斯分布
            nn.init.kaiming_normal(self.lora_a, a=0.01)
            # lora_b 初始化参数为 0
            self.lora_b = nn.Parameter(torch.zeros(rank, in_features))
            self.scale = lora_alpha / rank
            
            self.linear.weight.requires_grad = False
            self.linear.bias.requires_grad = False
        
        if merge:
            self.merge_weight()
            
    def merge_weight(self):
        if self.merge and self.rank > 0:
            self.linear.weight.data += self.scale * (self.lora_a @ self.lora_b)  
        
    def unmerge_weight(self):
        if self.rank > 0:
            self.linear.weight.data -= self.scale * (self.lora_a @ self.lora_b) 
            

    def forward(self, X):
        # X shape (batch_size, seq, in_features)
        # output shape (batch_size, seq, out_features)
        if self.rank > 0 and not self.merge:
            output = self.linear(X) + self.scale * (X @ (self.lora_a @ self.lora_b).T)
        elif self.rank > 0 and self.merge:
            output = self.linear(X)
        else:
            output = self.linear(X)
        
        return self.dropout(output)
            
        
if __name__ == "__main__":
    batch_size = 32
    seq_len = 128
    in_features = 768
    out_features = 512
    rank = 8
    lora_alpha = 16
    dropout = 0.1
    
    x = torch.randn(batch_size, seq_len, in_features)
    
    lora_layer = LinearLoRALayer(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        lora_alpha=lora_alpha,
        dropout=dropout,
        merge=False
    )
    output = lora_layer(x)
    print("output shape: ", output.shape)
    
    lora_layer_merge = LinearLoRALayer(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        lora_alpha=lora_alpha,
        dropout=dropout,
        merge=True
    ) 
    output_merge = lora_layer_merge(x)
    print("output_merge shape: ", output_merge.shape)  
    
    
    no_lora_layer = LinearLoRALayer(
        in_features=in_features,
        out_features=out_features,
        rank=-1,
        lora_alpha=lora_alpha,
        dropout=dropout,
        merge=False
    ) 
    output_no_lora = no_lora_layer(x)
    print("output_no_lora shape: ", output_no_lora.shape)  
    
    print("output and output_merge diff:")
    print(torch.max(torch.abs(output - output_merge)).item())    
    
    
    print("output and no_merge diff:")
    lora_layer_merge.unmerge_weight()
    output_no_merge = lora_layer_merge(x)
    print(torch.max(torch.abs(output - output_no_merge)).item())
    
    
    print("output and origin diff:")
    print(torch.max(torch.abs(output - output_no_lora)).item())    
    
    