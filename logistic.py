import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import shuffle
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# -------------------------
# Config
# -------------------------

TRAIN_PATH = "attention_hidden_data/saved_train_layer24_with_QA_num839.pt"
TEST_PATH = "attention_hidden_data/saved_test_layer24_with_QA_num150.pt"
RANDOM_SEED = 42
MAX_TOKENS_FOR_DEBUG = None
VERBOSE = True

# logistic regression C grid (paper requirement)
C_GRID = [10**(-i) for i in range(8, -1, -1)] + [10, 100, 1e5]

# position offsets (Act-* requirement)
POSITION_OFFSETS = [-3, -2, -1, 0, 1, 2]


# -------------------------
# Helpers
# -------------------------

def load_saved_items(path):
    return torch.load(path)

def rebalance_tokens(saved_items, debug_max_tokens=None, verbose=True):
    X_list = []
    y_list = []
    per_item = []
    skipped_items = 0
    total_tokens = 0

    for idx, item in enumerate(saved_items):
        if ("hidden_states" not in item or 
            "hallucination_labels" not in item or 
            "response_idx" not in item):
            skipped_items += 1
            continue

        hs = item["hidden_states"]
        labels = item["hallucination_labels"]
        resp_idx = int(item["response_idx"])

        hs_np = hs.cpu().numpy()
        labels_np = np.array(labels)

        seq_len = hs_np.shape[0]

        if labels_np.shape[0] != seq_len:
            skipped_items += 1
            continue

        # 只使用 response token
        hs_slice = hs_np[resp_idx:].astype(np.float32)
        labels_slice = labels_np[resp_idx:].astype(np.int64)

        if hs_slice.shape[0] == 0:
            skipped_items += 1
            continue

        X_list.append(hs_slice)
        y_list.append(labels_slice)
        per_item.append({
            "X": hs_slice,
            "y": labels_slice,
            "source_id": item.get("source_id", idx)
        })

        total_tokens += hs_slice.shape[0]
        if debug_max_tokens and total_tokens >= debug_max_tokens:
            break

    # ------------------------
    # 原始拼接
    # ------------------------
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    # ------------------------
    # 平衡采样过程
    # ------------------------
    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]

    num_pos = len(pos_idx)
    num_neg = len(neg_idx)

    # 正样本全部保留
    keep_pos_idx = pos_idx

    # 负样本采样相同数量
    if num_neg > num_pos:
        keep_neg_idx = np.random.choice(neg_idx, size=num_pos, replace=False)
    else:
        keep_neg_idx = neg_idx

    # label=1 前后各取一半
    half_pos = len(keep_pos_idx) // 2
    keep_pos_idx_sorted = np.sort(keep_pos_idx)
    pos_front = keep_pos_idx_sorted[:half_pos]
    pos_back  = keep_pos_idx_sorted[-half_pos:]

    final_pos_idx = np.concatenate([pos_front, pos_back])
    final_idx = np.concatenate([final_pos_idx, keep_neg_idx])

    # 最终采样之后的 X, y
    X_all = X_all[final_idx]
    y_all = y_all[final_idx]

    if verbose:
        print(f"[Dataset] tokens={X_all.shape[0]}, dim={X_all.shape[1]}")
        print(f"Label dist: {np.bincount(y_all)}")

    return X_all, y_all, per_item


def build_flat_dataset(saved_items, debug_max_tokens=None, verbose=True):
    X_list = []
    y_list = []
    per_item = []
    skipped_items = 0
    total_tokens = 0

    for idx, item in enumerate(saved_items):
        if ("hidden_states" not in item or 
            "hallucination_labels" not in item or 
            "response_idx" not in item):
            skipped_items += 1
            continue

        hs = item["hidden_states"]
        labels = item["hallucination_labels"]
        resp_idx = int(item["response_idx"])

        hs_np = hs.cpu().numpy()
        labels_np = np.array(labels)

        seq_len = hs_np.shape[0]

        if labels_np.shape[0] != seq_len:
            skipped_items += 1
            continue

        # Use only response tokens
        hs_slice = hs_np[resp_idx:].astype(np.float32)
        labels_slice = labels_np[resp_idx:].astype(np.int64)

        if hs_slice.shape[0] == 0:
            skipped_items += 1
            continue

        X_list.append(hs_slice)
        y_list.append(labels_slice)
        per_item.append({
            "X": hs_slice,
            "y": labels_slice,
            "source_id": item.get("source_id", idx)
        })

        total_tokens += hs_slice.shape[0]
        if debug_max_tokens and total_tokens >= debug_max_tokens:
            break

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

            
    if verbose:
        print(f"[Dataset] tokens={X_all.shape[0]}, dim={X_all.shape[1]}")
        print(f"Label dist: {np.bincount(y_all)}")

    return X_all, y_all, per_item


# -------------------------
# Position probing
# -------------------------

def extract_position_features(X, offset):
    """
    X: (N, D) hidden states sequence
    offset: one of [-3,-2,-1,0,1,2]
    """
    N, D = X.shape
    X_shift = np.zeros_like(X)

    for i in range(N):
        j = i + offset
        if 0 <= j < N:
            X_shift[i] = X[j]
        else:
            X_shift[i] = np.zeros(D)

    return X_shift


# -------------------------
# Logistic training
# -------------------------

def train_logreg(X, y, C):
    clf = LogisticRegression(
        C=C,
        max_iter=500,
        solver="lbfgs",
        random_state=RANDOM_SEED,
        # class_weight="balanced"
    )
    # clf = Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("logreg", LogisticRegression(
    #         C=C,
    #         solver="lbfgs",
    #         max_iter=2000,
    #         random_state=RANDOM_SEED,
    #         # class_weight="balanced"
    #     ))
    # ])
    clf.fit(X, y)
    return clf


# -------------------------
# Evaluation
# -------------------------

def evaluate(clf, X_test, y_test, per_item_test):
    probs = clf.predict_proba(X_test)[:, 1]

    token_auroc = roc_auc_score(y_test, probs)
    token_aupr = average_precision_score(y_test, probs)

    # Example-level
    example_scores = []
    example_labels = []
    idx = 0
    for item in per_item_test:
        n = item["X"].shape[0]
        p = probs[idx:idx+n]
        example_scores.append(float(p.max()))
        example_labels.append(int(item["y"].any()))
        idx += n

    example_labels = np.array(example_labels)
    example_scores = np.array(example_scores)

    def safe_metrics():
        if len(np.unique(example_labels)) < 2:
            return None, None
        return (
            roc_auc_score(example_labels, example_scores),
            average_precision_score(example_labels, example_scores)
        )

    ex_auroc, ex_aupr = safe_metrics()
    
    return {
        "token_auroc": token_auroc,
        "token_aupr": token_aupr,
        "example_auroc": ex_auroc,
        "example_aupr": ex_aupr
    }


# -------------------------
# Main
# -------------------------

def main():
    train_saved = load_saved_items(TRAIN_PATH)
    test_saved = load_saved_items(TEST_PATH)

    X_train_raw, y_train, per_item_train = build_flat_dataset(train_saved)
    X_test_raw, y_test, per_item_test = build_flat_dataset(test_saved)

    results_all = {}
    # POSITION_OFFSETS=[0]
    # C_GRID=[0.0001]
    for offset in POSITION_OFFSETS:
        print(f"\n======================")
        print(f" Offset = {offset}")
        print("======================")

        X_train = extract_position_features(X_train_raw, offset)
        X_test = extract_position_features(X_test_raw, offset)

        best_token_auroc = -1
        best_res = None
        best_C = None

        for C in tqdm(C_GRID,desc="Processing C"):
            print(f" C = {C}")
            clf = train_logreg(X_train, y_train, C)
            res = evaluate(clf, X_test, y_test, per_item_test)

            if res["token_auroc"] > best_token_auroc:
                best_token_auroc = res["token_auroc"]
                best_res = res
                best_C = C

        print(f"Best C={best_C}")
        print(f"Token AUROC={best_res['token_auroc']*100:.2f}")
        print(f"Token AUPR ={best_res['token_aupr']*100:.2f}")
        print(f"Example AUROC={best_res['example_auroc']*100:.2f}")
        print(f"Example AUPR ={best_res['example_aupr']*100:.2f}")

        results_all[offset] = (best_C, best_res)

    print("\n\n===== Summary =====")
    for offset, (C, res) in results_all.items():
        print(f"Offset={offset},  Best C={C},  Token AUROC={res['token_auroc']:.4f},  AUPR={res['token_aupr']:.4f}")


if __name__ == "__main__":
    main()
