import os

# TRAIN_DIR = "attributed_graphs_fixed/RAGtruth/Mistral-7B-Instruct-v0.3/train_layer24_Summary"
# TRAIN_DIR = "hypergraphs_cpu_mp/RAGtruth/Mistral-7B-Instruct-v0.3/train_QA/"

# files = [f for f in os.listdir(TRAIN_DIR) if f.endswith(".pt")]

# sizes = []
# for f in files:
#     path = os.path.join(TRAIN_DIR, f)
#     sizes.append(os.path.getsize(path))

# avg_size = sum(sizes) / len(sizes) / (1024 * 1024)

# print(f"Total files: {len(sizes)}")
# print(f"Average size: {avg_size:.2f} MB")


# import os
# import torch
# from tqdm import tqdm

# # 你的输出目录
# # graph_dir = "attributed_graphs_fixed/RAGtruth/Mistral-7B-Instruct-v0.3/train_layer24_QA"
# graph_dir = "hypergraphs_cpu_mp/RAGtruth/Mistral-7B-Instruct-v0.3/train_head1_QA/"

# files = sorted([f for f in os.listdir(graph_dir) if f.endswith(".pt")])

# print(f"检测到 {len(files)} 个图文件\n")

# count_bian=0
# for fname in tqdm(files):
#     path = os.path.join(graph_dir, fname)
#     data = torch.load(path)

#     #属性图
#     # edge_index = data["edge_index"]   # shape = [2, num_edges]
#     # num_edges = edge_index.shape[1]

#     #超图
#     num_edges = data["he_attr"].shape[0]
#     count_bian+=num_edges
#     # print(f"{fname}: 边数量 = {num_edges}")
# print(f"平均边数量 = {count_bian/len(files)}")


import os
import torch
import numpy as np
from tqdm import tqdm
import csv

# 修改你自己的目录
# graph_dir = "hypergraphs_cpu_mp/RAGtruth/Mistral-7B-Instruct-v0.3/train_head1_QA"
graph_dir = "attributed_graphs_fixed/RAGtruth/Mistral-7B-Instruct-v0.3/train_layer24_QA"

files = sorted([f for f in os.listdir(graph_dir) if f.endswith(".pt")])
print(f"Found {len(files)} files.\n")

rows = []
prompt_counts = []
response_counts = []

for fname in tqdm(files):
    path = os.path.join(graph_dir, fname)

    try:
        obj = torch.load(path)
    except Exception as e:
        print(f"Failed to load {fname}: {e}")
        continue

    # === 1. 获取 seq_len ===
    if "x" not in obj:
        print(f"Warning: no x in {fname}, skip.")
        continue
    seq_len = obj["x"].shape[0]

    # === 2. 获取 response_idx ===
    if "response_idx" not in obj:
        print(f"Warning: no response_idx in {fname}, skip.")
        continue

    response_idx = int(obj["response_idx"])

    # === 3. prompt/response 长度 ===
    prompt_len = response_idx
    resp_len = seq_len - response_idx

    prompt_counts.append(prompt_len)
    response_counts.append(resp_len)
    if resp_len==11:
        print(obj['source_id'])
    rows.append([fname, seq_len, prompt_len, resp_len])

# === 汇总统计 ===
print("\n=== Aggregate Stats ===")
print(f"Files processed: {len(rows)}")
print(f"Total prompt tokens: {sum(prompt_counts)}")
print(f"Total response tokens: {sum(response_counts)}")

if prompt_counts:
    print(f"Prompt per-file: mean={np.mean(prompt_counts):.2f}, median={np.median(prompt_counts):.2f}, min={min(prompt_counts)}, max={max(prompt_counts)}")

if response_counts:
    print(f"Response per-file: mean={np.mean(response_counts):.2f}, median={np.median(response_counts):.2f}, min={min(response_counts)}, max={max(response_counts)}")






# import os
# import torch
# from tqdm import tqdm

# # ====== 修改为你的超图目录 ======
# hypergraph_dir = "hypergraphs_cpu_mp/RAGtruth/Mistral-7B-Instruct-v0.3/test_Summary"

# # ====== 修改为你的模型配置 ======
# # 这里你需要填入真实的 L（层数），H（每层多少头）
# # 例如 Llama/Mistral 等通常是 L=32, H=32
# L = 32
# H = 32
# num_heads = L * H

# # ====== 扁平 head → (layer_id, head_id_in_layer) ======
# def decode_head_id(flat_head, H):
#     layer = flat_head // H
#     head  = flat_head % H
#     return int(layer), int(head)


# files = sorted([f for f in os.listdir(hypergraph_dir) if f.endswith(".pt")])
# print(f"Found {len(files)} hypergraph files.\n")

# for fname in files:
#     path = os.path.join(hypergraph_dir, fname)
#     data = torch.load(path, weights_only=False)

#     # ====== (1) 超边数量 ======
#     num_hyperedges = data["he_attr"].shape[0]

#     # 如果没有超边
#     if num_hyperedges == 0:
#         print(f"{fname}: hyperedges = 0, max_head = None")
#         continue

#     # ====== (2) 从 he_attr 的第3维恢复 normalized_head ======
#     normalized_heads = data["he_attr"][:, 2]  # shape [E]

#     # ====== (3) 恢复 flat head index ======
#     flat_heads = torch.round(normalized_heads * (num_heads - 1)).to(torch.int64)

#     max_flat_head = int(flat_heads.max().item())

#     # ====== (4) flat head → (layer_id, head_id_in_layer) ======
#     layer_id, head_id = decode_head_id(max_flat_head, H)

#     # ====== 打印结果 ======
#     print(
#         f"{fname}: hyperedges = {num_hyperedges}, "
#         f"max_flat_head = {max_flat_head} (layer={layer_id}, head={head_id})"
#     )


