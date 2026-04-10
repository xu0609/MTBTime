import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalDiscoveryModule(nn.Module):
    """
    Learnable time-varying causal adjacency module, supporting sparsity and gating
    """
    def __init__(self, num_vars, embed_dim=32, in_dim=None, top_k=10):
        super().__init__()
        self.num_vars = num_vars
        self.embed_dim = embed_dim
        self.in_dim = in_dim or num_vars
        self.var_emb = nn.Parameter(torch.randn(num_vars, embed_dim))
        self.context_mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )
        self.top_k = top_k

    def forward(self, context_vec):
        """
        context_vec: [batch, in_dim]
        ： [batch, num_vars, num_vars]
        """
        batch_size = context_vec.shape[0]
        c = self.context_mlp(context_vec)  # [batch, embed_dim]
        sim_i = torch.einsum('be,ie->bie', c, self.var_emb)
        sim_j = torch.einsum('be,je->bje', c, self.var_emb)
        adj = torch.einsum('bie,bje->bij', sim_i, sim_j)  # [batch, num_vars, num_vars]
        adj = torch.sigmoid(adj)  #
        #
        if self.top_k > 0:
            topk = torch.topk(adj, self.top_k, dim=-1)
            mask = torch.zeros_like(adj)
            mask.scatter_(-1, topk.indices, 1.0)
            adj = adj * mask
        #
        gate = self.gate_mlp(c).view(batch_size, 1, 1)  # [batch, 1, 1]
        adj = adj * gate
        return adj
