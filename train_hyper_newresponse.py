# hypergraph_hallucination.py
"""
Complete PyTorch + PyG script for response-level hallucination detection on hypergraphs.

Expected input: .pt files where each item is a dict with keys similar to your pipeline:
    - "x": node features (Tensor [N, F])
    - "he_incidence_index": incidence index (2 x M_total_members) like [node_indices; he_indices]
    - "he_attr": hyperedge attributes (Tensor [E, HeF])  (E = number of hyperedges)
    - "he_mark": per-member mark vector? (Tensor [M_total_members, mark_dim]) OR [E, mark_dim] (script supports [E, d])
    - "he_member_counts": Tensor [E] number of members per hyperedge
    - "y_token": token-level labels (used to compute graph-level label)
    - "response_idx": scalar or tensor indicating response start index (int)
You can adapt graph_to_data() if your .pt format differs.

Requirements: torch, torch_geometric, scikit-learn
"""

import os
import copy
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------- deterministic ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- config switches ----------------
USE_RESPONSE_ONLY = False      # False -> use context tokens (before response start); True -> use response tokens
POOL_TYPE = "attn"            # "attn" or "mean" or "pma" (pma not implemented here)
CORR_WEIGHT = 0.0             # set >0 to enable decorrelation loss
CLIP_GRAD_NORM = 1.0

# ---------------- utilities ----------------
def make_mlp(in_dim, hidden_dims: List[int], out_dim, activation=nn.ReLU, dropout=0.0):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LayerNorm(h))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def pearson_corr_scalar(a: torch.Tensor, b: torch.Tensor, eps=1e-8):
    """
    Differentiable Pearson correlation scalar (returns tensor scalar).
    Both a and b are 1D tensors on same device.
    """
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    a_mean = torch.mean(a)
    b_mean = torch.mean(b)
    a_cent = a - a_mean
    b_cent = b - b_mean
    cov = torch.mean(a_cent * b_cent)
    denom = (torch.sqrt(torch.mean(a_cent * a_cent) * torch.mean(b_cent * b_cent)) + eps)
    return cov / denom


# ---------------- Hypergraph message passing layer ----------------
class HypergraphLayer(nn.Module):
    """
    Simple hypergraph message passing:
      node -> hyperedge (aggregate members) -> hyperedge transform -> node (distribute)
    Implementation assumes:
      he_index: torch.LongTensor shape [2, M] where first row = node_idx, second row = he_idx (0..E-1)
      he_count: torch.FloatTensor shape [E] (members per hyperedge)
    """

    def __init__(self, node_dim, he_attr_dim, hidden_dim, residual=True):
        super().__init__()
        self.residual = residual
        # node->edge projector (per-member message)
        self.node2edge = make_mlp(node_dim + 0, [hidden_dim], hidden_dim)  # no member-specific features used here
        # if you want to use he_attr, include it in edge2node
        self.edge2node = make_mlp(he_attr_dim + hidden_dim, [hidden_dim], node_dim)
        self.ln = nn.LayerNorm(node_dim)

    def forward(self, x: torch.Tensor, he_index: torch.Tensor, he_attr: torch.Tensor, he_count: torch.Tensor):
        """
        x: [N, Dn]
        he_index: [2, M] long (node_idx, he_idx)
        he_attr: [E, Dha]
        he_count: [E]
        returns: updated node features [N, Dn]
        """
        device = x.device
        node_ids = he_index[0]    # shape [M]
        he_ids = he_index[1]      # shape [M]

        # ---- node -> hyperedge: compute member messages and aggregate per hyperedge ----
        member_msgs = self.node2edge(x[node_ids])  # [M, H]
        E = he_attr.size(0)
        agg_e = torch.zeros((E, member_msgs.size(-1)), device=device)  # [E, H]
        # sum into hyperedges
        agg_e.index_add_(0, he_ids, member_msgs)
        # normalize by member counts
        counts = he_count.view(-1, 1).to(device)
        agg_e = agg_e / (counts + 1e-6)

        # ---- edge -> node: for each member, compute incoming message from its hyperedge ----
        # gather per-member aggregated edge representation
        per_member_edge_repr = agg_e[he_ids]  # [M, H]
        # combine he_attr (per-edge) and per-member edge repr -> compute inc_msg per member
        # we need a per-member he_attr: pick he_attr[he_ids]
        per_member_he_attr = he_attr[he_ids] if he_attr.numel() > 0 else torch.zeros((per_member_edge_repr.size(0), 0), device=device)
        inc_in = torch.cat([per_member_he_attr, per_member_edge_repr], dim=-1)  # [M, Dha + H]
        inc_msg = self.edge2node(inc_in)  # [M, Dn]
        inc_msg = F.relu(inc_msg)

        # accumulate incoming messages to nodes
        N = x.size(0)
        out = torch.zeros_like(x)
        out.index_add_(0, node_ids, inc_msg)
        # normalize by node degree (#incident hyperedges per node)
        node_deg = torch.bincount(node_ids, minlength=N).float().unsqueeze(-1).to(device)
        out = out / (node_deg + 1e-6)

        out = self.ln(out)
        return x + out if self.residual else out


# ---------------- Model ----------------
class HypergraphHallucinationModel(nn.Module):
    def __init__(self,
                 node_dim,
                 he_attr_dim,
                 hidden_dim=128,
                 num_layers=2,
                 pool_type: str = "attn",
                 dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            HypergraphLayer(hidden_dim, he_attr_dim, hidden_dim, residual=True) for _ in range(num_layers)
        ])
        self.pool_type = pool_type
        if pool_type == "attn":
            self.att_q = nn.Linear(hidden_dim, hidden_dim)
            self.att_k = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        # placeholders for decorrelation
        self.last_graph_embed = None  # [G, D]
        self.last_resp_counts = None  # [G]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data, response_only=True):
        """
        data: batched PyG Data with fields:
            x [N, F]
            he_index [2, M] - LONG
            he_attr [E, Dha]
            he_count [E]
            node_pos [N]  (0..N-1 positions in sequence)
            response_idx scalar or Tensor [G] (response start idx)
            batch [N] graph index
        returns logits [G] (one score per graph/response)
        """
        x = data.x.to(torch.float32)
        h = F.relu(self.in_proj(x))  # [N, H]
        for layer in self.layers:
            h = layer(h, data.he_index.to(torch.long), data.he_attr.to(torch.float32), data.he_count.to(torch.float32))

        batch_idx = data.batch  # [N]
        num_graphs = int(batch_idx.max().item()) + 1

        # ---------- response-only mask ----------
        if response_only:
            if not torch.is_tensor(data.response_idx):
                response_idx = torch.tensor(data.response_idx, device=h.device)
            else:
                response_idx = data.response_idx.to(h.device)

            node_pos = data.node_pos.to(h.device)  # [N]
            if USE_RESPONSE_ONLY:
                mask = node_pos >= response_idx[batch_idx]
            else:
                mask = node_pos < response_idx[batch_idx]
            if mask.sum().item() == 0:
                # fallback: use all nodes if mask empty
                masked_h = h
                masked_batch = batch_idx
            else:
                masked_h = h[mask]
                masked_batch = batch_idx[mask]
        else:
            masked_h = h
            masked_batch = batch_idx

        # ---------- pooling ----------
        # We'll compute graph embeddings per graph based on masked nodes
        if self.pool_type == "mean":
            h_graph = torch.zeros((num_graphs, h.size(1)), device=h.device)
            h_graph.index_add_(0, masked_batch, masked_h)
            counts = torch.bincount(masked_batch, minlength=num_graphs).float().unsqueeze(1).to(h.device)
            denom = torch.sqrt(counts.clamp(min=1.0))
            h_graph = h_graph / denom
        elif self.pool_type == "attn":
            # attention pooling per token within each graph
            # compute attention scores per node: q(node) dot global_key(graph)
            # approximate per-graph key by mean of masked_h per graph
            D = masked_h.size(1)
            # compute per-graph mean
            h_mean = torch.zeros((num_graphs, D), device=h.device)
            h_mean.index_add_(0, masked_batch, masked_h)
            counts = torch.bincount(masked_batch, minlength=num_graphs).float().unsqueeze(1).to(h.device)
            h_mean = h_mean / (counts + 1e-6)

            # per-node scores: q(node) dot k(graph_of_node)
            q = self.att_q(masked_h)      # [M', H]
            k = self.att_k(h_mean)        # [G, H]
            # gather k per node
            k_per_node = k[masked_batch]  # [M', H]
            scores = (q * k_per_node).sum(dim=-1)  # [M']
            # softmax per graph: we need to exponentiate and normalize per graph
            exp_scores = torch.exp(scores - torch.max(scores))  # stable exp
            # sum exp per graph
            denom_per_graph = torch.zeros((num_graphs,), device=h.device)
            denom_per_graph.index_add_(0, masked_batch, exp_scores)
            denom_per_graph = denom_per_graph + 1e-8
            # attention weights per member
            att_w = exp_scores / denom_per_graph[masked_batch]  # [M']
            # weighted sum per graph
            h_graph = torch.zeros((num_graphs, D), device=h.device)
            h_graph.index_add_(0, masked_batch, masked_h * att_w.unsqueeze(-1))
        else:
            raise ValueError(f"Unknown pool_type {self.pool_type}")

        # store for decorrelation/reg
        self.last_graph_embed = h_graph  # [G, D]
        self.last_resp_counts = counts.squeeze(1)  # [G]

        logits = self.classifier(h_graph).view(-1)  # [G]
        return logits


# ---------------- dataset helpers ----------------
def graph_to_data(g: dict) -> Data:
    """
    Convert your saved graph dict to PyG Data.
    This is modeled closely to your earlier pipeline but you may need to adapt field names.
    """
    x = g["x"].clone().detach().to(torch.float32)
    he_index = g["he_incidence_index"].clone().detach().to(torch.long)  # shape [2, M]
    he_attr = g["he_attr"].clone().detach().to(torch.float32) if "he_attr" in g else torch.zeros((0, 1))
    he_mark = g.get("he_mark", None)
    he_count = g["he_member_counts"].clone().detach().to(torch.float32) if "he_member_counts" in g else torch.ones((he_attr.shape[0],), dtype=torch.float32)

    # graph-level label: if any token labeled positive => graph label 1
    y_token = g.get("y_token", None)
    if y_token is None:
        y = torch.tensor(0.0)
    else:
        y = torch.tensor(0.0) if y_token.sum() == 0 else torch.tensor(1.0)

    node_pos = torch.arange(x.size(0), dtype=torch.long)
    response_idx = torch.tensor(g["response_idx"], dtype=torch.long) if "response_idx" in g else torch.tensor(x.size(0) // 2, dtype=torch.long)

    data = Data(
        x=x,
        he_index=he_index,
        he_attr=he_attr,
        he_count=he_count,
        y=y,
        node_pos=node_pos,
        response_idx=response_idx
    )
    return data


def load_graph_dir(path: str) -> List[dict]:
    files = sorted([f for f in os.listdir(path) if f.endswith(".pt")])
    out = []
    for f in files:
        obj = torch.load(os.path.join(path, f))
        out.append(obj)
    return out


# ---------------- evaluation/training ----------------
def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            prob = torch.sigmoid(logits)
            ys.append(batch.y.cpu().numpy())
            ps.append(prob.cpu().numpy())

    if len(ys) == 0:
        return {"auroc": 0.5, "aupr": 0.0}

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)

    try:
        auroc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else 0.5
    except Exception:
        auroc = 0.5
    try:
        aupr = average_precision_score(ys, ps) if len(np.unique(ys)) > 1 else 0.0
    except Exception:
        aupr = 0.0

    return {"auroc": auroc, "aupr": aupr}


def compute_pos_weight_response(data_source):
    # accept DataLoader or list/dataset
    if hasattr(data_source, "dataset"):
        iterable = data_source.dataset
    else:
        iterable = data_source
    ys = []
    for d in iterable:
        yv = d.y.item() if torch.is_tensor(d.y) else float(d.y)
        ys.append(yv)
    ys = torch.tensor(ys, dtype=torch.float32)
    pos = (ys == 1).sum()
    neg = (ys == 0).sum()
    return float((neg.float() / max(pos.float(), 1.0)).item())


def train_model(train_loader: DataLoader, val_loader: DataLoader, node_dim, he_attr_dim, hp: dict):
    pos_weight_val = compute_pos_weight_response(train_loader)
    print(f"[Info] raw_pos_weight={pos_weight_val:.3f}")
    pos_weight_val = min(pos_weight_val, 50.0)
    print(f"[Info] clipped_pos_weight={pos_weight_val:.3f}")

    if pos_weight_val > 1.0:
        pos_weight_tensor = torch.tensor(pos_weight_val, device=DEVICE)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    model = HypergraphHallucinationModel(
        node_dim=node_dim,
        he_attr_dim=he_attr_dim,
        hidden_dim=hp.get("hidden_dim", 128),
        num_layers=hp.get("gnn_layers", 2),
        pool_type=hp.get("pool_type", POOL_TYPE),
        dropout=hp.get("dropout", 0.1)
    ).to(DEVICE)

    lr = float(hp.get("lr", 3e-4))
    opt = AdamW(model.parameters(), lr=lr, weight_decay=hp.get("weight_decay", 0.0))
    num_epochs = int(hp.get("epochs", 10))

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps) if total_steps > 0 else 0
    scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps) if total_steps > 0 else None

    best_val = -1.0
    best_state = None
    patience = int(hp.get("patience", 5))
    counter = 0
    corr_weight = float(hp.get("corr_weight", CORR_WEIGHT))

    for ep in range(num_epochs):
        model.train()
        total_loss = 0.0
        for bi, batch in enumerate(train_loader):
            batch = batch.to(DEVICE)
            logits = model(batch)  # [G]
            loss = loss_fn(logits, batch.y.to(DEVICE))

            # decorrelation loss (optional)
            if corr_weight > 0 and model.last_graph_embed is not None and model.last_resp_counts is not None:
                embed_scalar = model.last_graph_embed.norm(dim=1)  # [G]
                resp_counts = model.last_resp_counts.to(embed_scalar.device)
                corr = pearson_corr_scalar(embed_scalar, resp_counts)
                corr_loss = torch.abs(corr)
                loss = loss + corr_weight * corr_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
            opt.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

            # optional debug print
            if bi == 0 and (ep % 5 == 0):
                print(f"DEBUG epoch{ep+1} batch0 loss={loss.item():.4f} logits_shape={logits.shape} y_shape={batch.y.shape}")

        val_m = evaluate(model, val_loader)
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"[Epoch {ep+1}] loss={avg_loss:.4f}  AUROC={val_m['auroc']:.4f} AUPR={val_m['aupr']:.4f}")

        if val_m["aupr"] > best_val:
            best_val = val_m["aupr"]
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
    
def stratified_split_by_label_and_length(data_list, val_ratio=0.1, num_bins=5):

    # -----------------------
    # 1. 提取 labels 和 lengths
    # -----------------------
    labels = np.array([int(d.y.item()) for d in data_list])
    lengths = np.array([d.x.size(0) for d in data_list])  # 每个图的节点个数

    # -----------------------
    # 2. 根据长度分 quantile bins
    # -----------------------
    length_bins = np.quantile(lengths, q=np.linspace(0, 1, num_bins + 1))
    length_bins[-1] += 1e-6  # 防止边界点落不到最后一个bin

    length_bucket = np.digitize(lengths, length_bins)

    # -----------------------
    # 3. 联合分层标签：（label, length_bin）
    # -----------------------
    joint_label = labels.astype(str) + "_" + length_bucket.astype(str)

    # -----------------------
    # 4. 按 joint_label 进行分组
    # -----------------------
    groups = {}
    for i, g in enumerate(joint_label):
        groups.setdefault(g, []).append(i)

    # -----------------------
    # 5. 在每个 group 内部抽取验证集
    # -----------------------
    val_idx = []
    for key, idx_list in groups.items():
        group_size = len(idx_list)
        val_size = max(1, int(group_size * val_ratio))

        # 防止抽太多
        val_size = min(val_size, len(idx_list))

        sampled = random.sample(idx_list, val_size)
        val_idx.extend(sampled)

    val_idx = sorted(val_idx)
    all_idx = set(range(len(data_list)))
    train_idx = sorted(list(all_idx - set(val_idx)))

    # -----------------------
    # 6. 输出划分后的 Data 列表
    # -----------------------
    train_list = [data_list[i] for i in train_idx]
    val_list   = [data_list[i] for i in val_idx]

    return train_list, val_list, train_idx, val_idx

# ---------------- main (example usage) ----------------
if __name__ == "__main__":
    # --- USER ADJUSTABLE PATHS ---
    TRAIN_DIR = "hypergraphs_cpu_mp/RAGtruth/Mistral-7B-Instruct-v0.3/train_layer24_QA/"
    TEST_DIR  = "hypergraphs_cpu_mp/RAGtruth/Mistral-7B-Instruct-v0.3/test_layer24_QA/"

    # load raw dicts
    train_graphs = load_graph_dir(TRAIN_DIR)
    test_graphs = load_graph_dir(TEST_DIR)

    # convert to Data objects
    train_list = [graph_to_data(g) for g in train_graphs]
    test_list = [graph_to_data(g) for g in test_graphs]

    train_list, val_list, train_idx, val_idx = stratified_split_by_label_and_length(train_list)

    
    print(f"Loaded: train={len(train_list)} val={len(val_list)} test={len(test_list)}")

    # quick sanity: require at least one train sample
    if len(train_list) == 0:
        raise RuntimeError("No training samples found. Check TRAIN_DIR path and files.")

    # create DataLoaders (batch_size=1 per graph; you can increase if memory allows)
    train_loader = DataLoader(train_list, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=1, shuffle=False)

    # detect dims from first sample
    sample = train_list[0]
    node_dim = sample.x.shape[1]
    he_attr_dim = sample.he_attr.shape[1] if sample.he_attr.numel() > 0 else 1

    # training hyperparams
    HP = {
        "lr": 3e-4,
        "weight_decay": 0.0,
        "hidden_dim": 128,
        "gnn_layers": 2,
        "dropout": 0.2,
        "epochs": 30,
        "patience": 6,
        "pool_type": "attn",
        "corr_weight": 0.0,  # try 0.01 to decorrelate
    }

    model = train_model(train_loader, val_loader, node_dim, he_attr_dim, HP)

    # final test
    test_m = evaluate(model, test_loader)
    print("==== FINAL TEST METRICS ====")
    print(test_m)
