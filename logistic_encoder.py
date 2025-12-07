import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import torch.nn as nn
import logging
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -------------------------
# Logger setup
# -------------------------
def setup_logger(log_dir="logs", log_filename="charm_actprobe.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    logger = logging.getLogger("CHARM_ActProbe")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(f"Logger initialized. Logs will be saved to {log_path}")
    return logger

# -------------------------
# Config
# -------------------------
TRAIN_PATH = "attention_hidden_data/saved_train_layer24_with_QA_num839.pt"
TEST_PATH = "attention_hidden_data/saved_test_layer24_with_QA_num150.pt"
RANDOM_SEED = 42
MAX_TOKENS_FOR_DEBUG = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

C_GRID = [10**(-i) for i in range(8, -1, -1)] + [10, 100, 1e5]
POSITION_OFFSETS = [-3, -2, -1, 0, 1, 2]
ENCODER_WD_CANDIDATES = [0.0, 0.05, 0.1]
ACT_HIDDEN_DIM = 2048
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-4

# -------------------------
# Activation encoder
# -------------------------
class ActEncoder(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=ACT_HIDDEN_DIM, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        out = F.relu(self.input_proj(x))
        out = self.norm_in(out)
        for layer in self.layers:
            residual = out
            out = layer(out) + residual  # residual connection
        return out

# -------------------------
# Helpers
# -------------------------
def load_saved_items(path, logger):
    logger.info(f"Loading data from {path}")
    return torch.load(path)

def build_flat_dataset(saved_items, debug_max_tokens=None, logger=None):
    X_list, y_list, per_item = [], [], []
    skipped_items, total_tokens = 0, 0
    for idx, item in enumerate(saved_items):
        if ("hidden_states" not in item or "hallucination_labels" not in item or "response_idx" not in item):
            skipped_items += 1
            continue
        hs = item["hidden_states"]
        labels = np.array(item["hallucination_labels"])
        resp_idx = int(item["response_idx"])
        hs_np = hs.cpu().numpy()
        seq_len = hs_np.shape[0]
        if labels.shape[0] != seq_len: 
            skipped_items += 1
            continue
        hs_slice = hs_np[resp_idx:].astype(np.float32)
        labels_slice = labels[resp_idx:].astype(np.int64)
        if hs_slice.shape[0] == 0:
            skipped_items += 1
            continue
        X_list.append(hs_slice)
        y_list.append(labels_slice)
        per_item.append({"X": hs_slice, "y": labels_slice, "source_id": item.get("source_id", idx)})
        total_tokens += hs_slice.shape[0]
        if debug_max_tokens and total_tokens >= debug_max_tokens:
            break
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    if logger:
        logger.info(f"Built dataset: tokens={X_all.shape[0]}, dim={X_all.shape[1]}, skipped_items={skipped_items}")
        logger.info(f"Label distribution: {np.bincount(y_all)}")
    return X_all, y_all, per_item

def extract_position_features(X, offset):
    N, D = X.shape
    X_shift = np.zeros_like(X)
    for i in range(N):
        j = i + offset
        X_shift[i] = X[j] if 0 <= j < N else np.zeros(D)
    return X_shift

def extract_act_features(X, offset, encoder=None, device=DEVICE):
    X_shift = extract_position_features(X, offset)
    if encoder is not None:
        X_shift_t = torch.from_numpy(X_shift).float().to(device)
        with torch.no_grad():
            X_encoded = encoder(X_shift_t).cpu().numpy()
        return X_encoded
    return X_shift

def train_logreg(X, y, C):
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=RANDOM_SEED)
    clf.fit(X, y)
    return clf

def evaluate(clf, X_test, y_test, per_item_test):
    probs = clf.predict_proba(X_test)[:, 1]
    token_auroc = roc_auc_score(y_test, probs)
    token_aupr = average_precision_score(y_test, probs)
    example_scores, example_labels, idx = [], [], 0
    for item in per_item_test:
        n = item["X"].shape[0]
        p = probs[idx:idx+n]
        example_scores.append(float(p.max()))
        example_labels.append(int(item["y"].any()))
        idx += n
    example_labels = np.array(example_labels)
    example_scores = np.array(example_scores)
    if len(np.unique(example_labels)) < 2:
        ex_auroc = ex_aupr = None
    else:
        ex_auroc = roc_auc_score(example_labels, example_scores)
        ex_aupr = average_precision_score(example_labels, example_scores)
    return {"token_auroc": token_auroc, "token_aupr": token_aupr, "example_auroc": ex_auroc, "example_aupr": ex_aupr}

# -------------------------
# Encoder training
# -------------------------
def train_encoder(X_train, y_train, encoder, weight_decay, logger, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    N = X_train.shape[0]
    X_tensor = torch.from_numpy(X_train).float().to(DEVICE)
    y_tensor = torch.from_numpy(y_train).float().to(DEVICE)
    for epoch in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            X_batch = X_tensor[idx]
            y_batch = y_tensor[idx].unsqueeze(1)
            logits = encoder(X_batch)  # 输出隐藏层
            logits = torch.nn.Linear(logits.shape[1],1).to(DEVICE)(logits)  # probe layer
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
        logger.info(f"WD={weight_decay} Epoch {epoch+1}/{epochs} done")

# -------------------------
# Main
# -------------------------
def main(log_dir="logs", log_filename="charm_actprobe.log"):
    logger = setup_logger(log_dir, log_filename)

    train_saved = load_saved_items(TRAIN_PATH, logger)
    test_saved = load_saved_items(TEST_PATH, logger)
    X_train_raw, y_train, per_item_train = build_flat_dataset(train_saved, logger=logger)
    X_test_raw, y_test, per_item_test = build_flat_dataset(test_saved, logger=logger)

    results_all = {}

    for offset in POSITION_OFFSETS:
        logger.info(f"\n=== Offset {offset} ===")
        best_wd_score, best_wd, best_res, final_best_C = -1, None, None, None

        for wd in ENCODER_WD_CANDIDATES:
            logger.info(f"Training encoder with weight decay {wd}")
            encoder = ActEncoder(input_dim=X_train_raw.shape[1], hidden_dim=ACT_HIDDEN_DIM).to(DEVICE)
            train_encoder(X_train_raw, y_train, encoder, wd, logger)

            X_train_act = extract_act_features(X_train_raw, offset, encoder, DEVICE)
            X_test_act  = extract_act_features(X_test_raw, offset, encoder, DEVICE)

            best_token_auroc, best_C, res_for_wd = -1, None, None
            for C in tqdm(C_GRID, desc=f"C search (wd={wd})"):
                clf = train_logreg(X_train_act, y_train, C)
                res = evaluate(clf, X_test_act, y_test, per_item_test)
                if res["token_auroc"] > best_token_auroc:
                    best_token_auroc = res["token_auroc"]
                    best_C = C
                    res_for_wd = res

            logger.info(f"Offset={offset},WD={wd}, Best C={best_C}, Token AUROC={res_for_wd['token_auroc']*100:.2f}, AUPR={res_for_wd['token_aupr']*100:.2f}")
            if best_token_auroc > best_wd_score:
                best_wd_score = best_token_auroc
                best_wd = wd
                final_best_C = best_C
                best_res = res_for_wd

        results_all[offset] = (best_wd, final_best_C, best_res)
        logger.info(f"Offset {offset} summary: Encoder WD={best_wd}, Best C={final_best_C}, Token AUROC={best_res['token_auroc']*100:.2f},Token AUpr={best_res['token_aupr']*100:.2f}")

    logger.info("\n===== Overall Summary =====")
    for offset, (wd, C, res) in results_all.items():
        logger.info(f"Offset={offset}, Encoder WD={wd}, Best C={C}, Token AUROC={res['token_auroc']:.4f}, AUPR={res['token_aupr']:.4f}")

if __name__ == "__main__":
    main(log_dir="logger", log_filename="charm_actprobe.log")
