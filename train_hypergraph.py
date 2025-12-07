# run_hypercharm_stable.py (稳定版)
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
            nn.Linear(hp["hidden_dim"], max(8, hp["hidden_dim"] // 2)),
            nn.LayerNorm(max(8, hp["hidden_dim"] // 2)),
            nn.ReLU(),
            nn.Dropout(hp.get("dropout", 0.25)),
            nn.Linear(max(8, hp["hidden_dim"] // 2), 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        h = F.relu(self.in_proj(data.x))
        for layer in self.layers:
            h = layer(h, data.he_index, data.he_attr, data.he_mark, data.he_count)
        return self.pred(h).view(-1)


# ----------------- convert dict -> PyG Data -----------------
def graph_to_data(g):
    # x = torch.tensor(g["x"], dtype=torch.float32)
    # he_index = torch.tensor(g["he_incidence_index"], dtype=torch.long)
    # he_attr = torch.tensor(g["he_attr"], dtype=torch.float32)
    # he_mark = torch.tensor(g["he_mark"], dtype=torch.float32)
    # he_count = torch.tensor(g["he_member_counts"], dtype=torch.float32)
    # y = torch.tensor(g["y_token"], dtype=torch.float32)
    
    x = g["x"].clone().detach().to(torch.float32)
    he_index = g["he_incidence_index"].clone().detach().to(torch.long)
    he_attr = g["he_attr"].clone().detach().to(torch.float32)
    he_mark = g["he_mark"].clone().detach().to(torch.float32)
    he_count = g["he_member_counts"].clone().detach().to(torch.float32)
    y = g["y_token"].clone().detach().to(torch.float32)
    node_pos = torch.arange(x.size(0))
    response_idx = torch.tensor(g["response_idx"], dtype=torch.long)

    # ---- clamp + normalize x ----
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
            mask = batch.node_pos >= batch.response_idx[batch.batch]
            ys.append(batch.y[mask].cpu().numpy())
            ps.append(prob[mask].cpu().numpy())

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
# 1) 重新计算 pos_weight：只考虑 response 区间的 token
def compute_pos_weight_on_response(dataset):
    ys = []
    for d in dataset:
        mask = d.node_pos >= d.response_idx
        ys.append(d.y[mask])
    y = torch.cat(ys)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return neg.float() / max(pos.float(), 1.0)


# ----------------- train_model with Focal Loss -----------------
def train_model(train_loader, val_loader, node_dim, hedge_dim, hp):
    # 2) 训练里这么写：
    pos_weight_val = float(compute_pos_weight_on_response(train_loader.dataset))
    print(f"[Info] raw_pos_weight={pos_weight_val:.3f}")
    pos_weight_val = min(pos_weight_val, 10.0)  # 可以给个上限，比如 10
    print(f"[Info] clipped_pos_weight={pos_weight_val:.3f}")
    
    pos_weight_tensor = torch.tensor(pos_weight_val, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    model = HyperCHARM(node_dim, hedge_dim, hp).to(DEVICE)

    lr = float(hp.get("lr", 3e-4))  # 提高初始 lr
    opt = AdamW(model.parameters(), lr=lr, weight_decay=hp.get("weight_decay", 0.0))
    num_epochs = int(hp.get("epochs", 10))

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    # loss_fn = FocalLoss(alpha=pos_weight_val, gamma=2.0)  # focal loss

    best_val = -1
    best_state = None
    patience = 5
    counter = 0

    for ep in range(num_epochs):
        model.train()
        total_loss = 0.0
        for bi, batch in enumerate(train_loader):
            batch = batch.to(DEVICE)
            logits = model(batch)

            mask = batch.node_pos >= batch.response_idx[batch.batch]
            if mask.sum() == 0:
                continue

            loss = loss_fn(logits[mask], batch.y[mask])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
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
    TRAIN_DIR = "hypergraphs_cpu_mp/RAGtruth/Mistral-7B-Instruct-v0.3/train_head1_Summary/"
    TEST_DIR  = "hypergraphs_cpu_mp/RAGtruth/Mistral-7B-Instruct-v0.3/test_head1_Summary/"

    train_graphs = load_graph_dir(TRAIN_DIR)
    test_graphs  = load_graph_dir(TEST_DIR)

    train_list = [graph_to_data(g) for g in train_graphs]
    test_list  = [graph_to_data(g) for g in test_graphs]

    # random.shuffle(train_list)
    # val_size = max(1, int(len(train_list) * 0.1))
    # val_list = train_list[:val_size]
    # train_list = train_list[val_size:]
    # 统计正负样本索引
    pos_idx = [i for i, g in enumerate(train_list) if g.y.sum() > 0]  # 至少一个正样本
    neg_idx = [i for i, g in enumerate(train_list) if g.y.sum() == 0]
    
    val_size = max(1, int(len(train_list) * 0.1))
    
    # 按比例抽取正负样本
    num_pos_val = int(len(pos_idx) / len(train_list) * val_size)
    num_neg_val = val_size - num_pos_val
    
    import random
    random.seed(0)#42
    val_pos_idx = random.sample(pos_idx, min(num_pos_val, len(pos_idx)))
    val_neg_idx = random.sample(neg_idx, min(num_neg_val, len(neg_idx)))
    
    val_idx = val_pos_idx + val_neg_idx
    train_idx = list(set(range(len(train_list))) - set(val_idx))
    
    # 划分
    val_list = [train_list[i] for i in val_idx]
    train_list = [train_list[i] for i in train_idx]

    if len(train_list) == 0:
        raise RuntimeError("No training samples after split.")

    sample = train_list[0]
    node_dim = sample.x.shape[1]
    hedge_dim = sample.he_attr.shape[1] if sample.he_attr.numel() > 0 else 2

    SEARCH_SPACE = {
        "lr": [3e-4],
        "scheduler": ["cosine"],
        "dropout": [0.25],
        "hidden_dim": [128],
        "gnn_layers": [2],
        "weight_decay": [0.001],
        "residual_mp": [True],
        "epochs": [50]
    }

    grid_search(train_list, val_list, test_list, node_dim, hedge_dim, SEARCH_SPACE)
