# processed_graphs_fixed.py
import torch
import os
from tqdm import tqdm
from multiprocessing import Pool, set_start_method

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ===== 输入目录：你保存 sample_*.pt 的路径 =====
input_dir = "attention_hidden_data/RAGtruth/Mistral-7B-Instruct-v0.3/train_layer24_Summary"

# ===== 输出目录 =====
out_dir = "attributed_graphs_fixed/RAGtruth/Mistral-7B-Instruct-v0.3/train_layer24_Summary"
os.makedirs(out_dir, exist_ok=True)

USE_ONLY_SELF_ATT = False
tau = 0.05

# # ===== 遍历所有 sample pt 文件 =====
# files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pt")])

# print(f"共检测到 {len(files)} 个样本，开始图构建...")

# for fname in tqdm(files, desc="Processing samples"):
def process_single_file(fname):
    sample_path = os.path.join(input_dir, fname)

    # ----- 逐条加载 -----
    sample = torch.load(sample_path)

    source_id = sample["source_id"]
    token_ids = sample["token_ids"]
    # hidden_states = sample["hidden_states"].float()
    attention = sample["attention"].float()
    hallucination_labels = sample["hallucination_labels"]
    response_idx = sample["response_idx"]
    # V = sample.get("V", None)

    # ===== attention shape =====
    L, H, seq_len, seq_len2 = attention.shape
    assert seq_len == seq_len2

    # ===== flatten attention: (L*H, seq, seq) =====
    attention_flat = attention.reshape(L * H, seq_len, seq_len)

    # ====================================================
    # 1. 节点特征 X
    # ====================================================
    # 自注意力对角线：每层每头一个值 → (seq, L*H)
    self_att = torch.stack(
        [
            attention_flat[l, torch.arange(seq_len), torch.arange(seq_len)]
            for l in range(L * H)
        ],
        dim=1
    )
    X_V = self_att
    # if USE_ONLY_SELF_ATT:
    #     X_V = self_att
    # else:
    #     X_V = torch.cat([self_att, hidden_states], dim=1).float()

    # ====================================================
    # 2. 构造注意力边
    # ====================================================
    src_list, dst_list = [], []
    edge_attrs, edge_marks = [], []

    for i in range(seq_len):
        for j in range(i):  # causal: j → i

            # prompt→prompt 不要
            if j < response_idx and i < response_idx:
                continue

            a_ij = attention_flat[:, i, j]  # (L*H,)
            a_masked = a_ij.clone()
            a_masked[a_masked <= tau] = 0.0

            if torch.any(a_masked > 0):
                src_list.append(j)
                dst_list.append(i)
                edge_attrs.append(a_masked)

                # 边类型
                if j < response_idx and i >= response_idx:
                    mark = torch.tensor([1.0, 0.0])  # prompt→response
                else:
                    mark = torch.tensor([0.0, 1.0])  # response→response

                edge_marks.append(mark)

    # tensor 化
    if len(edge_attrs) > 0:
        edge_attr = torch.stack(edge_attrs, dim=0).float()
        edge_mark = torch.stack(edge_marks, dim=0).float()
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_attr = torch.zeros((0, L * H), dtype=torch.float32)
        edge_mark = torch.zeros((0, 2), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # labels
    y_token = torch.tensor(hallucination_labels, dtype=torch.long)

    # ===== 打包 =====
    graph_obj = {
        "source_id": source_id,
        "response_idx": response_idx,
        "token_ids": token_ids,
        # "V": V,
        "x": X_V,                # 节点特征
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_mark": edge_mark,
        "y_token": y_token,
    }

    # ===== 单独保存每个图 =====
    out_file = os.path.join(out_dir, f"graph_{fname}")
    torch.save(graph_obj, out_file)

if __name__ == "__main__":
    # spawn safe multiprocessing
    set_start_method("spawn", force=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".pt")]
    NUM_WORKERS = min(8, os.cpu_count())  # 合理并发，不要过高以免 I/O 抢占

    with Pool(processes=NUM_WORKERS) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_file, files), total=len(files)):
            pass

    print("Preprocessing Done!")
