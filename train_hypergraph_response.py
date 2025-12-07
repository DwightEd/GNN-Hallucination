# run_hypercharm_stable.py  (length-leakage mitigations: minimal edits)
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score

# ----------------- Deterministic / seeds -----------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Try to increase determinism (may slow down)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# try:
#     torch.use_deterministic_algorithms(True)
# except Exception:
#     pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- Utility -----------------
def make_mlp(in_dim, hidden_dims, out_dim, activation=nn.ReLU, dropout=0.0):
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

class PMA(nn.Module):
    def __init__(self, dim, k=1, heads=4):
        super().__init__()
        self.k = k
        # deterministic initialization using global seed above
        self.seed = nn.Parameter(torch.randn(k, dim))
        # ensure dropout=0.0 for determinism
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, h, batch):
        # h: [N_total, D], batch: [N_total] graph ids
        num_graphs = int(batch.max().item()) + 1
        outputs = []
        for i in range(num_graphs):
            hi = h[batch == i]  # [Ni, D]
            if hi.numel() == 0:
                outputs.append(torch.zeros((self.k, h.size(1)), device=h.device))
                continue
            hi = hi.unsqueeze(0)  # [1, Ni, D]
            si = self.seed.unsqueeze(0).to(h.device)  # [1, K, D]
            out, _ = self.mha(si, hi, hi)  # [1, K, D]
            out = self.norm1(out + si)
            out2 = self.ffn(out)
            out = self.norm2(out + out2)  # [1, K, D]
            outputs.append(out.squeeze(0))
        return torch.stack(outputs)  # [G, K, D]

# ----------------- HyperCharm (message passing) -----------------
class HyperCharm(MessagePassing):
    def __init__(self, node_dim, hedge_dim, hidden_dim, residual=True):
        super().__init__(aggr="add")
        self.residual = residual
        self.node2edge = make_mlp(node_dim + 2, [hidden_dim], hidden_dim)
        self.edge2node = make_mlp(hedge_dim + hidden_dim, [hidden_dim], node_dim)
        self.ln_out = nn.LayerNorm(node_dim)

    def forward(self, x, he_index, he_attr, he_mark, he_count):
        he_ids = he_index[1]
        node_ids = he_index[0]

        # ----- node -> edge -----
        msg_ne = self.node2edge(torch.cat([x[node_ids], he_mark[he_ids]], dim=-1))
        agg_e = torch.zeros((he_attr.size(0), msg_ne.size(-1)), device=x.device)
        # index_add_ is used; note: on GPU this can be non-deterministic in rare cases
        agg_e.index_add_(0, he_ids, msg_ne)
        agg_e = agg_e / (he_count.unsqueeze(-1) + 1e-6)

        # ----- edge -> node -----
        inc_msg = self.edge2node(torch.cat([he_attr[he_ids], agg_e[he_ids]], dim=-1))
        inc_msg = F.relu(inc_msg)

        out = torch.zeros_like(x)
        out.index_add_(0, node_ids, inc_msg)

        # normalize by node degree
        num_nodes = x.size(0)
        node_deg = torch.bincount(node_ids, minlength=num_nodes).float().unsqueeze(-1).to(x.device)
        out = out / (node_deg + 1e-6)

        out = self.ln_out(out)
        return x + out if self.residual else out


# ----------------- HyperCHARM model -----------------
class HyperCHARM(nn.Module):
    def __init__(self, node_dim, hedge_dim, hp):
        super().__init__()
        self.in_proj = nn.Linear(node_dim, hp["hidden_dim"])
        self.pma = PMA(hp["hidden_dim"])
        self.layers = nn.ModuleList([
            HyperCharm(
                node_dim=hp["hidden_dim"],
                hedge_dim=hedge_dim,
                hidden_dim=hp["hidden_dim"],
                residual=hp["residual_mp"]
            )
            for _ in range(hp["gnn_layers"])
        ])

        self.pred = nn.Sequential(
            nn.Linear(hp["hidden_dim"], hp["hidden_dim"] // 2),
            nn.ReLU(),
            nn.Dropout(hp.get("dropout", 0.1)),
            nn.Linear(hp["hidden_dim"] // 2, 1)
        )

        self._init_weights()
        # placeholders to expose last graph embed and response counts for decorrelation loss
        self.last_graph_embed = None  # [G, D]
        self.last_resp_counts = None  # [G]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data, pool="pma", response_only=True):
        h = F.relu(self.in_proj(data.x))
        for layer in self.layers:
            h = layer(h, data.he_index, data.he_attr, data.he_mark, data.he_count)

        batch_idx = data.batch                       # [N]
        num_graphs = int(batch_idx.max().item()) + 1

        # ========= response-only mask ==========
        if response_only:
            # response_idx could be scalar tensor or tensor [G]
            if not torch.is_tensor(data.response_idx):
                response_idx = torch.tensor(data.response_idx, device=h.device)
            else:
                response_idx = data.response_idx.to(h.device)

            node_pos = data.node_pos.to(h.device)  # [N]
            mask = node_pos >= response_idx[batch_idx]  # [N] boolean
            masked_h = h[mask]
            masked_batch = batch_idx[mask]
        else:
            masked_h = h
            masked_batch = batch_idx

        # ========= pooling with length-normalization ==========
        if pool in ["mean", "sum"]:
            # compute sum per graph (over masked nodes)
            h_graph = torch.zeros((num_graphs, h.size(1)), device=h.device)
            h_graph.index_add_(0, masked_batch, masked_h)  # sum
            counts = torch.bincount(masked_batch, minlength=num_graphs).float().unsqueeze(1).to(h.device)  # [G,1]
            # length-normalized sum: divide by sqrt(N) (reduces pure length signal)
            denom = torch.sqrt(counts.clamp(min=1.0))
            h_graph = h_graph / denom

            # store for decorrelation/reg
            self.last_graph_embed = h_graph  # [G, D]
            self.last_resp_counts = counts.squeeze(1)  # [G]

            # final projection
            return self.pred(h_graph).view(-1)

        # ========= PMA pooling (deterministic) ==========
        elif pool == "pma":
            # PMA returns [G, K, D]. We'll take mean over K heads, BUT we will also length-normalize
            # masked_h and masked_batch were computed above; PMA internally attends to all nodes in each graph.
            # PMA output is already scale-consistent, but to further reduce length leakage we apply a small normalization:
            h_graph_pma = self.pma(masked_h, masked_batch)  # [G, K, D]  (PMA implementation expects those args)
            h_graph = h_graph_pma.mean(dim=1)  # [G, D]

            # Heuristic length-normalization: divide embedding by sqrt(counts) as well
            counts = torch.bincount(masked_batch, minlength=num_graphs).float().unsqueeze(1).to(h.device)
            denom = torch.sqrt(counts.clamp(min=1.0))
            h_graph = h_graph / denom

            self.last_graph_embed = h_graph
            self.last_resp_counts = counts.squeeze(1)

            return self.pred(h_graph).view(-1)

        # ========= token-level prediction ==========
        else:
            # token-level predictions; no graph embedding stored
            self.last_graph_embed = None
            self.last_resp_counts = None
            return self.pred(h).view(-1)


# ----------------- convert dict -> PyG Data -----------------
def graph_to_data(g):
    x = g["x"].clone().detach().to(torch.float32)
    he_index = g["he_incidence_index"].clone().detach().to(torch.long)
    he_attr = g["he_attr"].clone().detach().to(torch.float32)
    he_mark = g["he_mark"].clone().detach().to(torch.float32)
    he_count = g["he_member_counts"].clone().detach().to(torch.float32)
    y = g["y_token"].clone().detach().to(torch.float32)

    # keep graph-level label (0/1): was intended behaviour in your pipeline
    y = torch.tensor(0.0) if y.sum() == 0 else torch.tensor(1.0)

    node_pos = torch.arange(x.size(0), dtype=torch.long)
    response_idx = torch.tensor(g["response_idx"], dtype=torch.long)

    # ---- clamp + normalize x (per-feature global style would be better, but keep minimal change) ----
    if x.numel() > 0:
        x = torch.clamp(x, -5.0, 5.0)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

    # ---- clamp + normalize he_attr ----
    if he_attr.numel() > 0:
        he_attr = torch.clamp(he_attr, 0.0, 1.0)
        he_attr = (he_attr - he_attr.mean(dim=0)) / (he_attr.std(dim=0) + 1e-6)

    return Data(
        x=x,
        he_index=he_index,
        he_attr=he_attr,
        he_mark=he_mark,
        he_count=he_count,
        y=y,
        node_pos=node_pos,
        response_idx=response_idx
    )


# ----------------- evaluate -----------------
def evaluate(model, loader):
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


# ----------------- helpers -----------------

def compute_pos_weight_response(dataset):
    ys = torch.tensor([d.y.item() for d in dataset])
    pos = (ys == 1).sum()
    neg = (ys == 0).sum()
    return neg.float() / max(pos.float(), 1.0)


def pearson_corr_scalar(a, b, eps=1e-8):
    """
    Compute Pearson correlation coefficient (scalar) between two 1D tensors a,b.
    Returns a single scalar value.
    """
    a = a.float()
    b = b.float()
    a_mean = a.mean()
    b_mean = b.mean()
    cov = ((a - a_mean) * (b - b_mean)).mean()
    denom = (a.std(unbiased=False) * b.std(unbiased=False)) + eps
    return cov / denom


# ----------------- train_model with decorrelation loss option -----------------
def train_model(train_loader, val_loader, node_dim, hedge_dim, hp):
    pos_weight_val = float(compute_pos_weight_response(train_loader.dataset))
    print(f"[Info] raw_pos_weight={pos_weight_val:.3f}")
    pos_weight_val = min(pos_weight_val, 15)
    print(f"[Info] clipped_pos_weight={pos_weight_val:.3f}")

    pos_weight_tensor = torch.tensor(pos_weight_val, device=DEVICE)

    if pos_weight_val > 3:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    model = HyperCHARM(node_dim, hedge_dim, hp).to(DEVICE)

    lr = float(hp.get("lr", 3e-4))
    opt = AdamW(model.parameters(), lr=lr, weight_decay=hp.get("weight_decay", 0.0))
    num_epochs = int(hp.get("epochs", 10))

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps) if total_steps > 0 else 0
    scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps) if total_steps>0 else None

    best_val = -1
    best_state = None
    patience = 5
    counter = 0

    corr_weight = float(hp.get("corr_weight", 0.0))  # weight for decorrelation loss (default 0 => off)

    for ep in range(num_epochs):
        model.train()
        total_loss = 0.0
        for bi, batch in enumerate(train_loader):
            batch = batch.to(DEVICE)
            logits = model(batch)  # forward sets model.last_graph_embed and last_resp_counts if graph-level

            loss = loss_fn(logits, batch.y)

            # ---- decorrelation loss: reduce correlation between graph embed norm and response length ----
            if corr_weight > 0 and model.last_graph_embed is not None and model.last_resp_counts is not None:
                # compute per-graph scalar: use L2-norm of embedding vector
                embed_norm = model.last_graph_embed.detach()  # [G, D]
                embed_scalar = embed_norm.norm(dim=1)  # [G]
                resp_counts = model.last_resp_counts  # [G]
                # pearson correlation
                corr = pearson_corr_scalar(embed_scalar, resp_counts)
                corr_loss = torch.abs(corr)  # absolute correlation to penalize both positive and negative correlation
                # Backprop through correlation requires embed_scalar not detached; use non-detach version:
                # recompute without detach to get gradient to model:
                embed_scalar2 = model.last_graph_embed.norm(dim=1)
                corr2 = pearson_corr_scalar(embed_scalar2, resp_counts)
                corr_loss = torch.abs(corr2)
                loss = loss + corr_weight * corr_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

        val_m = evaluate(model, val_loader)
        val_aupr = val_m["aupr"]
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"[Epoch {ep+1}] loss={avg_loss:.4f}  AUROC={val_m['auroc']:.4f} AUPR={val_aupr:.4f}")

        if val_aupr > best_val:
            best_val = val_aupr
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


# ----------------- grid search -----------------
def grid_search(train_list, val_list, test_list, node_dim, hedge_dim, space):
    keys = list(space.keys())
    vals = list(space.values())
    combos = list(__import__("itertools").product(*vals))

    train_loader = DataLoader(train_list, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=1)
    test_loader = DataLoader(test_list, batch_size=1)

    best_score = -1
    best_hp = None
    best_state = None

    for combo in combos:
        hp = dict(zip(keys, combo))
        print("\n===== Running HP:", hp)
        model = train_model(train_loader, val_loader, node_dim, hedge_dim, hp)
        val_m = evaluate(model, val_loader)
        aupr = val_m["aupr"]
        if aupr > best_score:
            best_score = aupr
            best_hp = hp
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        best_model = HyperCHARM(node_dim, hedge_dim, best_hp).to(DEVICE)
        best_model.load_state_dict(best_state)
        test_m = evaluate(best_model, test_loader)
    else:
        test_m = {"auroc": 0.5, "aupr": 0.0}

    print("\n===== BEST TEST =====")
    print(test_m)
    return best_hp, best_score, test_m


# ----------------- main -----------------
def load_graph_dir(path):
    files = sorted(f for f in os.listdir(path) if f.endswith(".pt"))
    return [torch.load(os.path.join(path, f)) for f in files]


if __name__ == "__main__":
    TRAIN_DIR = "hypergraphs_cpu_mp/RAGtruth/Mistral-7B-Instruct-v0.3/train_layer24_QA/"
    TEST_DIR  = "hypergraphs_cpu_mp/RAGtruth/Mistral-7B-Instruct-v0.3/test_layer24_QA/"

    train_graphs = load_graph_dir(TRAIN_DIR)
    test_graphs  = load_graph_dir(TEST_DIR)

    train_list = [graph_to_data(g) for g in train_graphs]
    test_list  = [graph_to_data(g) for g in test_graphs]

    # Prepare stratified val split by pos fraction (keeps pos ratio)
    pos_idx = [i for i, g in enumerate(train_list) if g.y.sum() > 0]
    neg_idx = [i for i, g in enumerate(train_list) if g.y.sum() == 0]

    val_size = max(1, int(len(train_list) * 0.1))
    num_pos_val = int(len(pos_idx) / max(len(train_list),1) * val_size)
    num_neg_val = val_size - num_pos_val

    val_pos_idx = random.sample(pos_idx, min(num_pos_val, len(pos_idx)))
    val_neg_idx = random.sample(neg_idx, min(num_neg_val, len(neg_idx)))

    val_idx = val_pos_idx + val_neg_idx
    train_idx = list(set(range(len(train_list))) - set(val_idx))

    val_list = [train_list[i] for i in val_idx]
    train_list = [train_list[i] for i in train_idx]

    if len(train_list) == 0:
        raise RuntimeError("No training samples after split.")

    sample = train_list[0]
    node_dim = sample.x.shape[1]
    hedge_dim = sample.he_attr.shape[1] if sample.he_attr.numel() > 0 else 3

    SEARCH_SPACE = {
        "lr": [3e-4],
        "scheduler": ["cosine"],
        "dropout": [0.25],
        "hidden_dim": [128],
        "gnn_layers": [2],
        "weight_decay": [0.001],
        "residual_mp": [True],
        "epochs": [50],
        # minimal new hyperparam: corr_weight (0 => off)
        "corr_weight": [0.01]
    }
    
    # quick length-only baseline on train 只考虑长度时的检测效果
    def collect_stats(list_data):
        seq_lens = []
        resp_lens = []
        node_counts = []
        he_counts = []
        labels = []
    
        for d in list_data:
            # 节点数
            seq_len = int(d.x.shape[0])
            seq_lens.append(seq_len)
    
            # response 长度
            resp_idx = int(d.response_idx.item())
            resp_lens.append(seq_len - resp_idx)
    
            # 节点数 = 序列长度
            node_counts.append(seq_len)
    
            # 超边数量（he_attr 行数 = 超边数）
            he_count = int(d.he_attr.shape[0]) if d.he_attr.numel() > 0 else 0
            he_counts.append(he_count)
    
            # 标签
            labels.append(int(d.y.item()))
    
        return (
            np.array(seq_lens),
            np.array(resp_lens),
            np.array(node_counts),
            np.array(he_counts),
            np.array(labels)
        )
    
    
    def eval_feature(feat, y):
        try:
            auroc = roc_auc_score(y, feat) if len(np.unique(y)) > 1 else 0.5
        except:
            auroc = 0.5
    
        try:
            aupr = average_precision_score(y, feat) if len(np.unique(y)) > 1 else 0.0
        except:
            aupr = 0.0
    
        return auroc, aupr
    
    
    # ---- 收集统计信息 ----
    seq_lens, resp_lens, node_counts, he_counts, labels = collect_stats(train_list)
    
    # ---- 逐特征测 AUROC / AUPR ----
    for name, feat in [
        ("seq_len", seq_lens),
        ("resp_len", resp_lens),
        ("he_count", he_counts)
    ]:
        a, p = eval_feature(feat, labels)
        print(f"{name}: AUROC={a*100:.2f}, AUPR={p*100:.2f}")


    grid_search(train_list, val_list, test_list, node_dim, hedge_dim, SEARCH_SPACE)
