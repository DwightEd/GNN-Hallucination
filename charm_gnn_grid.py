import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops


# ================================================================
# Message Passing Layer (Enhanced)
# Supports BatchNorm + Residual
# ================================================================
class CharmMessagePassing(MessagePassing):
    def __init__(self, in_node_dim, edge_dim, hidden_dim,
                 use_batchnorm=False, use_residual=False):
        super().__init__(aggr='add')

        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual

        # message MLP
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # update MLP
        self.up_mlp = nn.Sequential(
            nn.Linear(in_node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # BatchNorm
        if use_batchnorm:
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn = None

    def forward(self, x, edge_index, edge_attr):
        h_new = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

        # BatchNorm
        if self.bn is not None:
            h_new = self.bn(h_new)

        # Residual connection
        if self.use_residual:
            if h_new.size(-1) == x.size(-1):
                h_new = h_new + x  # identity residual
            else:
                # project x
                x_proj = F.linear(x, torch.eye(h_new.size(-1), device=x.device))
                h_new = h_new + x_proj

        return h_new

    def message(self, x_i, x_j, edge_attr):
        inp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_mlp(inp)

    def update(self, aggr_out, x):
        inp = torch.cat([x, aggr_out], dim=-1)
        return self.up_mlp(inp)


# ================================================================
# CHARM Model (Enhanced)
# Supports:
# - BatchNorm
# - Residual connections
# - Full compatibility with hyperparam search
# ================================================================
class CHARM(nn.Module):
    def __init__(self,
                 node_input_dim,
                 edge_attr_dim,
                 hidden_dim=256,
                 num_layers=3,
                 dropout=0.1,
                 token_output_dim=1,
                 graph_output_dim=1,
                 pool_type='mean',
                 use_batchnorm=False,
                 use_residual=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual

        # Node feature projection
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)

        # Edge projection (identity-like)
        self.edge_proj = nn.Linear(edge_attr_dim, edge_attr_dim)

        # Stack message passing layers
        self.layers = nn.ModuleList([
            CharmMessagePassing(
                in_node_dim=hidden_dim,
                edge_dim=edge_attr_dim,
                hidden_dim=hidden_dim,
                use_batchnorm=use_batchnorm,
                use_residual=use_residual
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Token-level head
        self.token_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, token_output_dim)
        )

        # Graph-level head
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, graph_output_dim)
        )

    def forward(self, data, token_level=True):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.node_proj(x)
        e = self.edge_proj(edge_attr)

        # GNN layers
        for layer in self.layers:
            h = layer(h, edge_index, e)
            h = self.dropout(h)

        # Node-level prediction
        token_logits = self.token_head(h).squeeze(-1)

        if token_level:
            return token_logits, h

        # Graph-level prediction (mean pooling)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        pooled = global_mean_pool(h, batch)
        graph_logits = self.graph_head(pooled).squeeze(-1)

        return graph_logits, pooled
