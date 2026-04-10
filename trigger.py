import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.nn.pytorch import GraphConv
import dgl


class GraphConvolutionLayer(nn.Module):
    def __init__(self, config,in_features, out_features,atk_vars, device='cuda'):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        self.atk_vars = atk_vars  #
        self.device = device


        self.causal_prior = None
        self.causal_prior_weight = float(getattr(config, 'causal_prior_weight', 0.3))  # 0~1，默认0.3

        max_k = len(atk_vars) * len(atk_vars)
        self.k = min(config.get('trigger_top_k', 10), max_k)
        """if self.k != config.trigger_top_k:
            print(f"Warning: Adjusted trigger_top_k from {config.trigger_top_k} to {self.k} (max possible)")"""

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        """
        :param x: the input features, shape: (batch_size, n, in_features)
        :param adj: the adjacency matrix, shape: (n, n)
        """
        support = torch.einsum("bnc,ck->bnk", x, self.weight)  # torch.bmm(x, self.weight)
        output = torch.einsum("mn,bnk->bmk", adj, support)  # torch.bmm(adj.unsqueeze(0), support)
        return output


class TgrGCN(nn.Module):
    def __init__(self, config, sim_feats, atk_vars, device='cuda'):
        super(TgrGCN, self).__init__()

        self.distance_threshold = nn.Parameter(torch.tensor(0.5))
        self.k = config.get('trigger_top_k', 10)
        self.input_dim = config.bef_tgr_len
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.trigger_len
        self.init_bound = config.epsilon

        self.constant_MLP = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)  # the first element is set to 0
        )
        self.structure_MLP = nn.Sequential(
             nn.Linear(sim_feats.shape[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 64)  # the first element is set to 0
        )

        self.conv1 = GraphConvolutionLayer(config, self.input_dim, self.hidden_dim, atk_vars, device)
        self.conv2 = GraphConvolutionLayer(config, self.hidden_dim, self.output_dim, atk_vars, device)

        self.sim_feats = torch.from_numpy(sim_feats).float().to(device)[atk_vars]  # (n, c)

        self.device = device
        self.layer_num = 2

        self.atk_vars = atk_vars
        self.causal_prior = None
        self.causal_prior_weight = float(getattr(config, 'causal_prior_weight', 0.3))

        for m in self.constant_MLP:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.uniform_(m.bias, -0.2, 0.2)

    def set_causal_prior(self, A_prior, weight=None):

        if hasattr(self, "graph_layer"):
            self.graph_layer.set_causal_prior(A_prior, weight)
        if hasattr(self, "gcn1"):
            self.gcn1.set_causal_prior(A_prior, weight)
        if hasattr(self, "gcn2"):
            self.gcn2.set_causal_prior(A_prior, weight)




    def forward(self, x, constant_alpha=0.5):
        n = self.sim_feats.shape[0]
        x = x.view(-1, n, self.input_dim).to(self.device)


        bias = self.constant_MLP(torch.zeros_like(x))
        bias = torch.tanh(bias) * self.init_bound * constant_alpha

        A = self.cal_structure()
        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
        A = torch.matmul(torch.matmul(D, A), D).to(self.device)

        # Dual channel perturbation enhancement
        h1 = self.conv1(x, A)
        h2 = self.conv1(x, A.T)  #
        h = F.relu(h1 + h2)  #

        perturb = self.conv2(h, A)
        perturb = torch.sigmoid(perturb) * self.init_bound


        gate = torch.sigmoid(perturb)  #
        out = gate * (perturb + bias) + (1 - gate) * x[..., -1:]
        return out, perturb + bias

    def cal_structure(self):
        node_num = self.sim_feats.shape[0]
        node_outs = self.structure_MLP(self.sim_feats.detach())  # (n, c)

        # Calculate similarity matrix
        A = F.cosine_similarity(
            node_outs.unsqueeze(0),
            node_outs.unsqueeze(1),
            dim=-1
        )
        A = torch.sigmoid(A - self.distance_threshold)

        # Integrating causal priors
        if self.causal_prior is not None:
            A_c = self.causal_prior.to(A.device)

            if A_c.shape != A.shape:
                raise ValueError(f"Causal prior shape {A_c.shape} != structure A {A.shape}")
            w = float(self.causal_prior_weight)

            A = (1.0 - w) * A + w * A_c



        max_possible_k = node_num * node_num
        safe_k = min(self.k, max_possible_k)

        # TOp-K sparsity
        if safe_k > 0:  #
            topk_mask = torch.zeros_like(A)
            topk_values, topk_indices = torch.topk(A.flatten(), safe_k)
            topk_mask.view(-1)[topk_indices] = 1
            A = A * topk_mask
        else:
            A = torch.zeros_like(A)

        # Self reinforcing loop
        identity = torch.eye(node_num).to(self.device)
        A = (1 - identity) * A + identity

        return A

    def set_causal_prior(self, A_prior: torch.Tensor, weight: float = None):
        if weight is not None:
            self.causal_prior_weight = float(weight)

        try:
            A_sub = self._align_prior_to_trigger_nodes(A_prior)
        except Exception as e:
            print(f"[TgrGCN] WARN: align causal prior failed: {e}")
            A_sub = None
        self.causal_prior = A_sub  #


    def _align_prior_to_trigger_nodes(self, A_prior: torch.Tensor):

        if A_prior is None:
            return None
        A_prior = A_prior.detach().clone()
        if self.atk_vars is None:
            return A_prior  # Return as is without subset information

        idx = self.atk_vars
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(idx, dtype=torch.long)
        else:
            idx = idx.to(dtype=torch.long)

        #
        if A_prior.size(0) <= int(idx.max()):
            raise ValueError(
                f"A_prior size {A_prior.size()} too small for indices up to {int(idx.max())}"
            )

        #
        A_sub = A_prior.index_select(0, idx).index_select(1, idx)
        return A_sub
