# run_charm_full_v3.py
# ---------------------------------------------------
# CHARM + MultiLayer Activation Encoder (full hyper param control)
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
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
import copy


# ================================
# ---------- CONFIG --------------
# ================================
ACT_DIM = 4096


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
# MultiLayer Activation Encoder + optional residual
# ---------------------------------------------------
class GLUBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, norm="layer", use_residual=True):
        super().__init__()
        self.use_residual = use_residual and (in_dim == out_dim)

        # 为了 GLU: Linear 输出 2 * out_dim
        self.fc = nn.Linear(in_dim, out_dim * 2)

        # Normalization
        if norm == "batch":
            self.norm = nn.BatchNorm1d(in_dim)
        elif norm == "layer":
            self.norm = nn.LayerNorm(in_dim)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)

        # GLU: split along last dimension
        a, g = self.fc(h).chunk(2, dim=-1)
        h = a * torch.sigmoid(g)

        h = self.dropout(h)

        if self.use_residual:
            return x + h
        return h


class MultiLayerActivationEncoder(nn.Module):
    def __init__(
        self,
        act_dim=4096,
        hidden_dims=[3072, 2048, 1024, 512],
        out_dim=256,
        dropout=0.1,
        norm="layer",
        use_residual=True
    ):
        super().__init__()

        layers = []
        prev = act_dim
        for h in hidden_dims:
            layers.append(
                GLUBlock(prev, h, dropout=dropout, norm=norm, use_residual=use_residual)
            )
            prev = h

        self.layers = nn.ModuleList(layers)
        self.out_layer = nn.Linear(prev, out_dim)

    def forward(self, act):
        x = act
        for layer in self.layers:
            x = layer(x)
        return self.out_layer(x)

# ---------------------------------------------------
# CHARM Message Passing + optional residual
# ---------------------------------------------------
class CharmMP(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim, residual=True):
        super().__init__(aggr="add")
        self.residual = residual
        self.msg_mlp = make_mlp(node_dim + edge_dim + 2, [hidden_dim], hidden_dim)
        self.up_mlp  = make_mlp(node_dim + hidden_dim, [hidden_dim], node_dim)

    def forward(self, x, edge_index, edge_attr, edge_mark, deg_out):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_mark=edge_mark, deg_out=deg_out)
        if self.residual:
            return x + out
        return out

    def message(self, x_j, edge_attr, edge_mark):
        return self.msg_mlp(torch.cat([x_j, edge_attr, edge_mark], dim=-1))

    def update(self, aggr_out, x, deg_out):
        deg = deg_out.clone()
        deg[deg == 0] = 1.0
        neigh_avg = aggr_out / deg.unsqueeze(-1)
        return self.up_mlp(torch.cat([x, neigh_avg], dim=-1))


# ---------------------------------------------------
# CHARM Model
# ---------------------------------------------------
class CHARM(nn.Module):
    def __init__(self, base_dim, edge_dim, act_dim, hp):
        super().__init__()

        # Activation encoder
        # self.act_encoder = MultiLayerActivationEncoder(
        #     act_dim=act_dim,
        #     hidden_dims=[2048, 1024, 512],
        #     out_dim=256,#out_dim=256
        #     dropout=hp["dropout"],
        #     norm=hp["norm"],
        #     use_residual=hp["residual_encoder"]
        # )

        # node_input_dim = base_dim + 256
        node_input_dim=base_dim
        
        self.in_proj = nn.Linear(node_input_dim, hp["hidden_dim"])

        # GNN layers
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
    
        

    def forward(self, data):
        base = data.x
        # act  = data.act

        # act_enc = self.act_encoder(act)
        # x = torch.cat([base, act_enc], dim=-1)

        # h = F.relu(self.in_proj(x))
        h = F.relu(self.in_proj(base))
        for mp in self.mp_layers:
            h = F.relu(mp(h, data.edge_index, data.edge_attr, data.edge_mark, data.deg_out))

        return self.pred(h).view(-1)


# ================================
# ---------- DATA UTILS ----------
# ================================
def graph_to_data(graph, act_dim):
    x_all = torch.tensor(graph["x"], dtype=torch.float32)
   
    base_dim = x_all.shape[1] - act_dim
    if base_dim>0:
        base = x_all[:, :base_dim]
        act  = x_all[:, base_dim:]
    else:
        base = x_all
        act=None

    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
    edge_attr  = torch.tensor(graph["edge_attr"], dtype=torch.float32)
    edge_mark  = torch.tensor(graph["edge_mark"], dtype=torch.float32)

    y = torch.tensor(graph["y_token"], dtype=torch.float32)

    N = base.shape[0]
    node_pos = torch.arange(N, dtype=torch.long)   
    response_idx = torch.tensor(graph["response_idx"], dtype=torch.long)

    N = base.shape[0]
    deg_out = torch.zeros(N, dtype=torch.float32)
    deg_out.index_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
    

    return Data(
        x=base, act=act,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_mark=edge_mark,
        y=y, deg_out=deg_out,
        node_pos=node_pos,
        response_idx=response_idx
    )


# ================================
# ---------- LOSS / EVAL ----------
# ================================
# def compute_pos_weight(dataset):
#     y = torch.cat([d.y.view(-1) for d in dataset])
#     pos = (y == 1).sum()
#     neg = (y == 0).sum()
#     return neg.float() / max(pos.float(), 1)
def compute_pos_weight(dataset):
    ys = []
    for d in dataset:
        mask = d.node_pos >= d.response_idx
        ys.append(d.y[mask])
    y = torch.cat(ys)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return neg.float() / max(pos.float(), 1.0)

def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            prob = torch.sigmoid(logits)

            node_pos = batch.node_pos
            response_idx_per_graph = batch.response_idx
            batch_idx = batch.batch  # per-node graph id

            # mask: only response nodes (same logic as training)
            mask = node_pos >= response_idx_per_graph[batch_idx]
            ys.append(batch.y[mask].cpu().numpy())
            ps.append(prob[mask].cpu().numpy())

    if len(ys) == 0:
        return {"auroc": 0.5, "aupr": 0.0}
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)

    auroc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else 0.5
    aupr = average_precision_score(ys, ps) if len(np.unique(ys)) > 1 else 0.0
    return {"auroc": auroc, "aupr": aupr}


def select_threshold(ys, ps):
    precision, recall, th = precision_recall_curve(ys, ps)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = np.argmax(f1)
    return 0.5 if best_idx >= len(th) else th[best_idx]

# ================================
# ---------- TRAINING ------------
# ================================

def train_model(train_loader, val_loader, base_dim, edge_dim, hp):
    
    # train_loader = DataLoader(train_list, batch_size=hp["batch_size"], shuffle=True)
    # val_loader   = DataLoader(val_list, batch_size=hp["batch_size"], shuffle=False)

    pos_weight = compute_pos_weight(train_loader.dataset).to(DEVICE)

    model = CHARM(base_dim, edge_dim, ACT_DIM, hp).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=hp["lr"],
        weight_decay=hp["weight_decay"]
    )

    # Scheduler setup
    num_epochs = hp.get("epochs", 10)              # 默认20，或由hp传入
    
    if hp["scheduler"] == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    
    elif hp["scheduler"] == "cosine":
        # Cosine Annealing with Warmup (手动实现 Warmup)
        num_training_steps = num_epochs * len(train_loader)
        num_warmup_steps = int(0.05 * num_training_steps)
        
        scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
    
    else:
        scheduler = None


    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # loss_fn = FocalLoss()

    

    best_state = None
    best_aupr = -1
    patience = 5           # 连续多少个 epoch 没提升就停止
    counter = 0            # 计数器
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
             # --------- 只计算 response 节点的 loss ---------
            y = batch.y.to(DEVICE) 
            node_pos = batch.node_pos
            batch_idx = batch.batch
            response_idx_per_graph = batch.response_idx
        
            mask = node_pos >= response_idx_per_graph[batch_idx]  # 只选择 response 节点
            masked_logits = logits[mask]
            masked_y = y[mask]
        
            loss = loss_fn(masked_logits, masked_y)
            # loss = loss_fn(logits, batch.y.to(DEVICE))#修改为只计算response loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if hp["scheduler"] == "cosine":
                scheduler.step()
                
            total_loss += loss.item()

        val_metrics = evaluate(model, val_loader)
        val_aupr = val_metrics["aupr"]
        avg_loss = total_loss / max(1, len(train_loader))
        # step scheduler
        if hp["scheduler"] == "reduce":
            scheduler.step(val_aupr)

        print(f"[Epoch {epoch+1}] loss={avg_loss:.4f}  val_AUROC={val_metrics['auroc']:.4f} val_AUPR={val_aupr:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")

        if val_aupr > best_aupr:
            best_aupr = val_aupr
            best_state = copy.deepcopy(model.state_dict())
            # best_state = model.state_dict()
            counter = 0  # 重置计数器
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break

    model.load_state_dict(best_state)
    return model

# ================================================
#          AUTOMATIC GRID SEARCH MODULE
# ================================================
import itertools
import json
import time

# ================================================
#          AUTOMATIC GRID SEARCH MODULE (v2)
#      now includes TEST evaluation for best HP
# ================================================
import itertools
import json
import time

def grid_search(
    train_list,
    val_list,
    test_list,
    base_dim,
    edge_dim,
    param_grid,
    start_run=None
):
    """
    param_grid: dict containing lists of values, e.g.:
        {
            "lr": [0.001, 0.0005],
            ...
        }
    test_list: list of PyG Data objects (test dataset)
    """

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    if start_run is not None:
        combos = combos[start_run:]

    print(f"\n===== GRID SEARCH START (total={len(combos)}) =====\n")

    results = []
    best_score = -1
    best_hp = None
    best_model_state = None
    
    train_loader = DataLoader(train_list, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=32, shuffle=False)
    for idx, combo in enumerate(combos):
        hp = dict(zip(keys, combo))

        print("\n---------------------------------------------")
        print(f"[RUN {idx+1}/{len(combos)}]  Hyperparams:")
        print(json.dumps(hp, indent=4))
        print("---------------------------------------------")
        
        # Train with these hyperparameters
        start_time = time.time()
        # train_loader = DataLoader(train_list, batch_size=hp["batch_size"], shuffle=True)
        # val_loader = DataLoader(val_list, batch_size=hp["batch_size"], shuffle=False)
        
        model = train_model(train_loader, val_loader, base_dim, edge_dim, hp)
        duration = time.time() - start_time

        # Evaluate on validation set
        
        val_metrics = evaluate(model, val_loader)
        val_aupr = val_metrics["aupr"]
        val_auroc = val_metrics["auroc"]
        print(f"→ VAL AUROC={val_auroc:.4f} AUPR = {val_aupr:.4f}, time = {duration:.1f}s")

        # Record every run
        results.append({
            "hp": hp,
            "val_aupr": float(val_aupr),
            "time": duration
        })

        # Update best
        if val_aupr > best_score:
            print("✓ New best HP!")
            best_score = val_aupr
            best_hp = hp
            best_model_state = model.state_dict()
            
        summary = {
            "index":idx,
            "hp": hp,
            "val_auroc":val_auroc,
            "val_aupr": val_aupr,

        }
        with open("grid_summary.jsonl", "a") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    # ======================================================
    #     After all runs — evaluate BEST model on TEST
    # ======================================================
    print("\n===== Evaluating BEST model on TEST set =====")
    best_model = CHARM(base_dim, edge_dim, ACT_DIM, best_hp).to(DEVICE)
    best_model.load_state_dict(best_model_state)

   
    test_metrics = evaluate(best_model, test_loader)

    print("\n===== FINAL TEST RESULTS =====")
    print("TEST AUROC:", test_metrics["auroc"])
    print("TEST AUPR :", test_metrics["aupr"])

    # Save results
    summary = {
        "best_hp": best_hp,
        "best_val_aupr": best_score,
        "test_auroc": float(test_metrics["auroc"]),
        "test_aupr": float(test_metrics["aupr"])
    }

    with open("grid_best_summary.json", "a") as f: 
        json.dump(summary, f, indent=4)
    

    torch.save({
        "model_state": best_model_state,
        "best_hp": best_hp,
        "test_metrics": test_metrics
    }, "grid_best_model.pt")

    print("\nSaved best model + summary to:")
    print(" - grid_best_summary.json")
    print(" - grid_best_model.pt")

    return best_hp, best_score, summary, results


# ================================
# ---------- MAIN ---------------
# ================================
if __name__ == "__main__":
    print("Loading datasets...")

    # ------------------------------
    # 新的数据加载：从目录读取多个 graph_xxx.pt 文件
    # ------------------------------
    def load_graph_dir(path):
        files = sorted([
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".pt")
        ])
        graphs = []
        for f in files:
            g = torch.load(f)
            graphs.append(g)
        return graphs

    TRAIN_DIR = "attributed_graphs_fixed/RAGtruth/Mistral-7B-Instruct-v0.3/train_layer24_Summary"
    TEST_DIR  = "attributed_graphs_fixed/RAGtruth/Mistral-7B-Instruct-v0.3/test_layer24_Summary"

    train_graphs = load_graph_dir(TRAIN_DIR)
    test_graphs  = load_graph_dir(TEST_DIR)
    

    train_list = [graph_to_data(g, ACT_DIM) for g in train_graphs]
    test_list  = [graph_to_data(g, ACT_DIM) for g in test_graphs]

    # validation split
    # val_size = max(1, int(len(train_list) * 0.1))#按比例抽取val
    # val_list = train_list[:val_size]
    # # val_list = test_list
    # train_list = train_list[val_size:]
    
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
    # print(f"base_dim:{base_dim}")
    edge_dim = sample.edge_attr.shape[1]

    # SEARCH_SPACE = {
    #     "lr": [0.001, 0.0005],
    #     "scheduler": ["cosine","reduce"],
    #     "batch_size": [32],
    #     "dropout": [0.25, 0.5],
    #     "hidden_dim": [32,64, 128],
    #     "gnn_layers": [1, 2, 3],
    #     "weight_decay": [0.0, 0.001],
    #     "norm": ["layer", "batch"],
    #     "residual_encoder": [True, False],
    #     "residual_mp": [True, False],
    #     "epochs": [50]
    # }
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
    auroc_results=[]
    auprc_results=[]
    for i in range(3):
        best_hp, best_val, summary, results = grid_search(
            train_list,
            val_list,
            test_list,
            base_dim,
            edge_dim,
            SEARCH_SPACE,
            start_run=None
        )
        auroc_results.append(summary['test_auroc'])
        auprc_results.append(summary['test_aupr'])
        
    print("All auroc:",auroc_results)
    print("Averge auroc:",sum(auroc_results) / len(auroc_results))
    print("All auprc:",auprc_results)
    print("Averge auprc:",sum(auprc_results) / len(auprc_results))
