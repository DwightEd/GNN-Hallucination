# run_charm_graph_level.py
# ---------------------------------------------------
# CHARM + MultiLayer Activation Encoder (graph-level prediction)
# ---------------------------------------------------

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import copy
from transformers import get_cosine_schedule_with_warmup
import random
import json
import time
import itertools


# ================================
# ---------- CONFIG --------------
# ================================
ACT_DIM = 4096

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------
# Hyperparameters (Grid Search Ready)
# ---------------------------------------------------
HP = {
    "lr": 0.001,
    "scheduler": "cosine",
    "batch_size": 32,
    "dropout": 0.25,
    "hidden_dim": 128,
    "gnn_layers": 2,
    "weight_decay": 0.0,
    "norm": "layer",
    "residual_encoder": True,
    "residual_mp": True,
    "epochs": 50,
}

# ================================
# ---------- MODULES -------------
# ================================

def make_mlp(in_dim, hidden_dims, out_dim, activation=nn.ReLU, dropout=0.0, norm=None):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if norm == "batch":
            layers.append(nn.BatchNorm1d(h))
        elif norm == "layer":
            layers.append(nn.LayerNorm(h))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------
# CHARM Message Passing + optional residual
# ---------------------------------------------------
class CharmMP(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim, residual=True):
        super().__init__(aggr="add")
        self.residual = residual
        self.msg_mlp = make_mlp(node_dim + edge_dim + 2, [hidden_dim], hidden_dim)
        self.up_mlp  = make_mlp(node_dim + hidden_dim, [hidden_dim], node_dim)

    def forward(self, x, edge_index, edge_attr, edge_mark, deg_in):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_mark=edge_mark, deg_in=deg_in)
        if self.residual:
            return x + out
        return out

    def message(self, x_j, edge_attr, edge_mark):
        return self.msg_mlp(torch.cat([x_j, edge_attr, edge_mark], dim=-1))

    def update(self, aggr_out, x, deg_in):
        deg = deg_in.clone()
        deg[deg == 0] = 1.0
        neigh_avg = aggr_out / deg.unsqueeze(-1)
        return self.up_mlp(torch.cat([x, neigh_avg], dim=-1))


# ---------------------------------------------------
# CHARM Model (Graph-level)
# ---------------------------------------------------
class CHARM(nn.Module):
    def __init__(self, base_dim, edge_dim, hp):
        super().__init__()

        node_input_dim = base_dim
        self.in_proj = nn.Linear(node_input_dim, hp["hidden_dim"])

        self.mp_layers = nn.ModuleList([
            CharmMP(hp["hidden_dim"], edge_dim, hp["hidden_dim"], residual=hp["residual_mp"])
            for _ in range(hp["gnn_layers"])
        ])

        self.pred = nn.Sequential(
            nn.Linear(hp["hidden_dim"], hp["hidden_dim"] // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hp["hidden_dim"] // 2, 1)
        )

    def forward(self, data, pool="mean", response_only=True):
        h = F.relu(self.in_proj(data.x))
        for mp in self.mp_layers:
            h = F.relu(mp(h, data.edge_index, data.edge_attr, data.edge_mark, data.deg_in))
    
        if pool in ["mean", "sum"]:
            batch_idx = data.batch
            num_graphs = batch_idx.max().item() + 1
    
            if response_only:
                node_pos = data.node_pos  # 节点在图中的索引
                response_idx = data.response_idx  # 每个图 response 开始位置
                mask = node_pos >= response_idx[batch_idx]  # 只选 response token
                masked_h = h[mask]
                masked_batch = batch_idx[mask]
            else:
                masked_h = h
                masked_batch = batch_idx
    
            # 初始化图级特征
            h_graph = torch.zeros((num_graphs, h.size(1)), device=h.device)
            h_graph.index_add_(0, masked_batch, masked_h)
    
            if pool == "mean":
                counts = torch.bincount(masked_batch).unsqueeze(1).float()
                counts[counts==0] = 1.0
                h_graph = h_graph / counts
    
            return self.pred(h_graph).view(-1)
    
        else:  # token-level
            return self.pred(h).view(-1)




# ================================
# ---------- DATA UTILS ----------
# ================================
def graph_to_data(graph, act_dim):
    # x_all = torch.tensor(graph["x"], dtype=torch.float32)
    x_all = graph["x"].clone().detach().to(torch.float32)
    base_dim = x_all.shape[1] - act_dim
    if base_dim>0:
        base = x_all[:, :base_dim]
        act  = x_all[:, base_dim:]
    else:
        base = x_all
        act=None
    edge_index = graph["edge_index"].clone().detach().to(torch.long)
    edge_attr = graph["edge_attr"].clone().detach().to(torch.float32)
    edge_mark = graph["edge_mark"].clone().detach().to(torch.float32)
    # edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
    # edge_attr  = torch.tensor(graph["edge_attr"], dtype=torch.float32)
    # edge_mark  = torch.tensor(graph["edge_mark"], dtype=torch.float32)

    y = graph["y_token"].clone().detach().to(torch.float32)
    # y = torch.tensor(graph["y_token"], dtype=torch.float32)
    y = torch.tensor(0.0) if y.sum() == 0 else torch.tensor(1.0)

    N = base.shape[0]

    # 节点位置索引
    node_pos = torch.arange(N, dtype=torch.long)

    # response token 的起始位置 (这里假设 graph["response_idx"] 存储了每个图 response 的开始节点)
    # response_idx = graph["response_idx"].clone().detach().to(torch.long)
    response_idx = torch.tensor(graph.get("response_idx", [0]*N), dtype=torch.long)
    

    deg_in = torch.zeros(N, dtype=torch.float32)
    if edge_index.numel() > 0:
        deg_in.index_add_(0, edge_index[0], torch.ones(edge_index.size(1)))

    return Data(
        x=base,
        act=act,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_mark=edge_mark,
        y=y,
        deg_in=deg_in,
        node_pos=node_pos,
        response_idx=response_idx
    )



# ================================
# ---------- LOSS / EVAL ----------
# ================================
def compute_pos_weight(dataset):
    y = torch.tensor([d.y for d in dataset])
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return neg.float() / max(pos.float(), 1)

def evaluate(model, loader):
    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            prob = torch.sigmoid(model(batch)).cpu().numpy()
            ys.append(batch.y.cpu().numpy())
            ps.append(prob)

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    return {
        "auroc": roc_auc_score(ys, ps),
        "aupr": average_precision_score(ys, ps),
        "fpr": roc_curve(ys, ps)[0],
        "tpr": roc_curve(ys, ps)[1],
        "precision": precision_recall_curve(ys, ps)[0],
        "recall": precision_recall_curve(ys, ps)[1],
        "ys": ys,
        "ps": ps,
    }

def select_threshold(ys, ps):
    precision, recall, th = precision_recall_curve(ys, ps)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = np.argmax(f1)
    return 0.5 if best_idx >= len(th) else th[best_idx]


# ================================
# ---------- TRAINING ------------
# ================================
def train_model(train_loader, val_loader, base_dim, edge_dim, hp):
    pos_weight = compute_pos_weight(train_loader.dataset).to(DEVICE)
    print(f"[Info] pos_weight={pos_weight:.3f}")
    model = CHARM(base_dim, edge_dim, hp).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=hp["lr"],
        weight_decay=hp["weight_decay"]
    )

    # Scheduler
    num_epochs = hp.get("epochs", 50)
    if hp["scheduler"] == "cosine":
        num_training_steps = num_epochs * len(train_loader)
        num_warmup_steps = int(0.05 * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        scheduler = None

    if pos_weight>3:
        pos_weight_tensor = torch.tensor(pos_weight, device=DEVICE)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    
    best_state = None
    best_aupr = -1
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            y = batch.y.to(DEVICE).view(-1)

            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if hp["scheduler"] == "cosine":
                scheduler.step()
            total_loss += loss.item()

        val_metrics = evaluate(model, val_loader)
        val_aupr = val_metrics["aupr"]
        print(f"[Epoch {epoch+1}] loss={total_loss:.4f}  val_AUROC={val_metrics['auroc']:.4f} val_AUPR={val_aupr:.4f}")

        if val_aupr > best_aupr:
            best_aupr = val_aupr
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


# ================================
# ---------- GRID SEARCH ----------
# ================================
def grid_search(train_list, val_list, test_list, base_dim, edge_dim, param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    print(f"\n===== GRID SEARCH START (total={len(combos)}) =====\n")
    results = []
    best_score = -1
    best_hp = None
    best_model_state = None

    train_loader = DataLoader(train_list, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_list, batch_size=32, shuffle=False)

    for idx, combo in enumerate(combos):
        hp = dict(zip(keys, combo))
        print(f"[RUN {idx+1}/{len(combos)}] Hyperparams: {hp}")

        model = train_model(train_loader, val_loader, base_dim, edge_dim, hp)
        val_metrics = evaluate(model, val_loader)
        val_aupr = val_metrics["aupr"]
        print(f"→ VAL AUPR={val_aupr:.4f}")

        results.append({"hp": hp, "val_aupr": float(val_aupr)})

        if val_aupr > best_score:
            best_score = val_aupr
            best_hp = hp
            best_model_state = model.state_dict()

    # 测试集
    print("\n===== Evaluating BEST model on TEST set =====")
    test_loader = DataLoader(test_list, batch_size=best_hp["batch_size"], shuffle=False)
    best_model = CHARM(base_dim, edge_dim, best_hp).to(DEVICE)
    best_model.load_state_dict(best_model_state)
    test_metrics = evaluate(best_model, test_loader)
    print("TEST AUROC:", test_metrics["auroc"])
    print("TEST AUPR :", test_metrics["aupr"])
    
     # Save results
    summary = {
        "best_hp": best_hp,
        "best_val_aupr": best_score,
        "test_auroc": float(test_metrics["auroc"]),
        "test_aupr": float(test_metrics["aupr"])
    }

    with open("response_grid_best_summary.json", "a") as f: 
        json.dump(summary, f, indent=4)
    # 保存
    torch.save({
        "model_state": best_model_state,
        "best_hp": best_hp,
        "test_metrics": test_metrics
    }, "response_grid_best_model_graph.pt")

    return best_hp, best_score, test_metrics, results

def compute_hallucination_ratio(dataset):
    y_list = []
    for data in dataset:
        y = data.y
        if y.dim() == 0:  # 标量情况
            y_list.append(y.item())
        else:  # token-level y 取是否存在 hallucination
            y_list.append(1.0 if y.sum() > 0 else 0.0)
    y_array = torch.tensor(y_list)
    total = len(y_array)
    halluc_count = (y_array == 1).sum().item()
    ratio = halluc_count / total
    return total, halluc_count, ratio
# ================================
# ---------- MAIN ---------------
# ================================
if __name__ == "__main__":
    TRAIN_DIR = "attributed_graphs_fixed/RAGtruth/Mistral-7B-Instruct-v0.3/train_layer24_Summary"
    TEST_DIR  = "attributed_graphs_fixed/RAGtruth/Mistral-7B-Instruct-v0.3/test_layer24_Summary"

    def load_graph_dir(path):
        files = sorted([os.path.join(path,f) for f in os.listdir(path) if f.endswith(".pt")])
        return [torch.load(f,weights_only=False) for f in files]

    train_graphs = load_graph_dir(TRAIN_DIR)
    test_graphs  = load_graph_dir(TEST_DIR)

    train_list = [graph_to_data(g, ACT_DIM) for g in train_graphs]
    test_list  = [graph_to_data(g, ACT_DIM) for g in test_graphs]
    
    total, halluc_count, ratio = compute_hallucination_ratio(test_list)
    print(f"总图数量: {total}, 幻觉图数量: {halluc_count}, 幻觉比例: {ratio:.4f}")
    
    # 统计正负样本索引
    pos_idx = [i for i, g in enumerate(train_list) if g.y.sum() > 0]  # 至少一个正样本
    neg_idx = [i for i, g in enumerate(train_list) if g.y.sum() == 0]
    
    val_size = max(1, int(len(train_list) * 0.1))
    
    # 按比例抽取正负样本
    num_pos_val = int(len(pos_idx) / len(train_list) * val_size)
    num_neg_val = val_size - num_pos_val
    
    import random
    random.seed(0)#QA:0
    val_pos_idx = random.sample(pos_idx, min(num_pos_val, len(pos_idx)))
    val_neg_idx = random.sample(neg_idx, min(num_neg_val, len(neg_idx)))
    
    val_idx = val_pos_idx + val_neg_idx
    train_idx = list(set(range(len(train_list))) - set(val_idx))
    
    # 划分
    val_list = [train_list[i] for i in val_idx]
    train_list = [train_list[i] for i in train_idx]

    sample = train_list[0]
    base_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1]

    SEARCH_SPACE = {
        "lr": [0.0005],
        "scheduler": ["cosine"],
        "batch_size": [32],
        "dropout": [0.25],
        "hidden_dim": [128],
        "gnn_layers": [3],
        "weight_decay": [0.001],
        "norm": ["layer"],
        "residual_encoder": [True],
        "residual_mp": [True],
        "epochs": [50]
    }

    # best_hp, best_val, test_metrics, results = grid_search(
    #     train_list, val_list, test_list, base_dim, edge_dim, SEARCH_SPACE
    # )
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    # inputs: train_list/test_list are lists of Data objects as in your script
    def collect_stats(list_data):
        seq_lens = []
        resp_lens = []
        node_counts = []
        he_counts = []
        labels = []
        for d in list_data:
            seq_len = int(d.x.shape[0])
            resp_idx = int(d.response_idx.item()) if torch.is_tensor(d.response_idx) else int(d.response_idx)
            seq_lens.append(seq_len)
            resp_lens.append(seq_len - resp_idx)
            node_counts.append(seq_len)
            # hyperedge count: he_attr len
            he_counts.append(int(d.he_attr.shape[0]) if d.he_attr.numel()>0 else 0)
            labels.append(int(d.y.item()))
        return np.array(seq_lens), np.array(resp_lens), np.array(node_counts), np.array(he_counts), np.array(labels)
    
    # example usage:
    seq_lens, resp_lens, node_counts, he_counts, labels = collect_stats(train_list)
    
    def eval_feature(feat, y):
        # use score = feat (higher -> positive); compute AUROC/AUPR
        try:
            auroc = roc_auc_score(y, feat) if len(np.unique(y))>1 else 0.5
        except:
            auroc = 0.5
        try:
            aupr = average_precision_score(y, feat) if len(np.unique(y))>1 else 0.0
        except:
            aupr = 0.0
        return auroc, aupr
    
    for name, feat in [("seq_len", seq_lens), ("resp_len", resp_lens), ("#he", he_counts)]:
        a, p = eval_feature(feat, labels)
        print(f"{name}: AUROC={a*100:.2f}, AUPR={p*100:.2f}")

