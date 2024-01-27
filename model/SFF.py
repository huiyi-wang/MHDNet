import torch
import torch.nn as nn
from functools import partial
from .HAM import Hierarchical_Attention
class Injector(nn.Module):
    def __init__(self, dim=768, reduce_dim=256, fs_size1=384, gamma=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=1.0, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = Hierarchical_Attention(dim_in=dim,reduce_dim=reduce_dim,Fs_size=fs_size1,gamma=gamma)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, feat):

        def _inner_forward(query, feat):
            # query.shape [64,193,768]
            # feat.shape  [64,1009,768]
            # query = self.query_norm(query)
            # feat  = self.query_norm(feat)
            attn1,attn2 = self.attn(query_feature=query,Input_feature=feat)
            return query + self.gamma * attn2

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)  # 显存优化
        else:
            query = _inner_forward(query, feat)

        return query