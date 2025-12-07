# charm_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops

class CharmMessagePassing(MessagePassing):
    def __init__(self, in_node_dim, edge_dim, hidden_dim):
        # we'll use 'add' (sum) aggregator as in paper
        super().__init__(aggr='add')
        # message MLP: inputs = [h_i, h_j, edge_attr, edge_mark] OR we can allow msg to ignore h_i/h_j
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_node_dim + in_node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # update MLP: inputs = [h_i, aggregated_msg]
        self.up_mlp = nn.Sequential(
            nn.Linear(in_node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        # x: (N, F), edge_index: (2, E), edge_attr: (E, D)
        # propagate will call message & aggregate & update
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: (E, F) receiver node features
        # x_j: (E, F) sender node features
        # edge_attr: (E, D)
        inp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_mlp(inp)

    def update(self, aggr_out, x):
        # aggr_out: (N, hidden_dim) aggregated messages for each receiver
        inp = torch.cat([x, aggr_out], dim=-1)
        return self.up_mlp(inp)


class CHARM(nn.Module):
    def __init__(self,
                 node_input_dim,
                 edge_attr_dim,
                 hidden_dim=256,
                 num_layers=3,
                 dropout=0.1,
                 token_output_dim=1,
                 graph_output_dim=1,
                 pool_type='mean'  # or 'sum'
                 ):
        super().__init__()
        self.node_input_dim = node_input_dim
        self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # initial linear to project input node features into hidden_dim
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)

        # optionally project edge_attr to a dimension
        self.edge_proj = nn.Linear(edge_attr_dim, edge_attr_dim)  # identity-ish; adjust if needed

        # stack of message passing layers
        self.layers = nn.ModuleList([
            CharmMessagePassing(hidden_dim, edge_attr_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # token-level prediction head (token-wise classification)
        self.token_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, token_output_dim)
        )

        # graph-level prediction head (after pooling)
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, graph_output_dim)
        )

    def forward(self, data, token_level=True):
        # data.x: (N, node_input_dim)
        # data.edge_index: (2, E)
        # data.edge_attr: (E, edge_attr_dim) - should include the mark if desired
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # project
        h = self.node_proj(x)   # (N, hidden_dim)
        e = self.edge_proj(edge_attr) if edge_attr is not None else None

        # message passing stack
        for layer in self.layers:
            h_new = layer(h, edge_index, e)
            h = self.dropout(h_new)  # residual could be added if desired

        # token-level logits
        token_logits = self.token_head(h).squeeze(-1)  # (N,) if token_output_dim==1

        if token_level:
            return token_logits, h  # return hidden per node optionally
        else:
            # graph-level pooling: require data.batch for batching; if single graph, set batch = zeros(N)
            batch = getattr(data, 'batch', None)
            if batch is None:
                batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            pooled = global_mean_pool(h, batch)  # (num_graphs, hidden_dim)
            graph_logits = self.graph_head(pooled).squeeze(-1)
            return graph_logits, pooled
