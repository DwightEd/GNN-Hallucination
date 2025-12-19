# build_hypergraphs_stable.py
import torch
import os
from tqdm import tqdm
from multiprocessing import Pool, set_start_method

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ----- 修改：你的输入/输出 路径按需调整 -----
input_dir = "attention_hidden_data/RAGtruth/Mistral-7B-Instruct-v0.3/test_layer24_Summary"
out_dir = "hypergraphs_cpu_mp/RAGtruth/Mistral-7B-Instruct-v0.3/test_head1_Summary"
os.makedirs(out_dir, exist_ok=True)

# ----- 超参改动（更稳健） -----
tau = 0.05              # 提高阈值，减少超边数量（减少极端聚合）

def process_single_file(fname):
    path = os.path.join(input_dir, fname)
    if os.path.exists(os.path.join(out_dir, f"hypergraph_{fname}")):
        return fname
    sample = torch.load(path, weights_only=False)

    source_id = sample["source_id"]
    token_ids = sample["token_ids"]
    hallucination_labels = sample["hallucination_labels"]
    response_idx = int(sample["response_idx"])

    attention = sample["attention"].float()   # [L, H, seq_len, seq_len]
    L, H, seq_len, _ = attention.shape
    num_heads = L * H
    attention_flat = attention.reshape(num_heads, seq_len, seq_len)

    # ---------- 节点特征：保留原始 self-att，统一交给 graph_to_data 标准化 ----------
    diag = torch.arange(seq_len)
    self_att = attention_flat[:, diag, diag].transpose(0, 1)  # [seq_len, num_heads]
    self_att = torch.clamp(self_att, 0.0, 1.0)  # 保证在 [0,1]，后面再标准化
    X_V = self_att

    # ---------- 掩码 ----------
    tri_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=-1)
    prompt_mask = torch.arange(seq_len) < response_idx

    # 只禁用 prompt 内部的注意力，其余保持下三角
    valid_mask = tri_mask.clone()
    valid_mask[:response_idx, :response_idx] = False

    he_node_list = []
    he_id_list = []
    he_attr_list = []
    he_mark_list = []
    he_member_counts_list = []
    he_counter = 0

    # 一些可调超参
    MIN_MEMBERS_IN_HE = 2           # 至少两个节点才建超边
    MAX_MEMBERS_PER_HE = 10**9         # 单条超边最多成员
    MAX_HE_PER_SAMPLE = 10**9
    # MAX_MEMBERS_PER_HE = 32         # 单条超边最多成员
    # MAX_HE_PER_SAMPLE = 20000
    TOPK_PER_ROW = 16               # 每个 (head, i) 行最多保留多少成员
    INCLUDE_CENTER_TOKEN = True     # 是否把中心 i 也加入超边成员
    USE_TOPK_FALLBACK = True        # 如果 > tau 的成员太少，是否用 top-k 兜底

    for head in range(num_heads):
        att = attention_flat[head]  # [seq_len, seq_len]

        # 预先过滤一次无效位置
        att_masked = att * valid_mask  # 无效位置为 0

        for i in range(seq_len):
            # 只对 response 段建超边（这句可选，其实 valid_mask 已经保证 i<prompt 没有有效位置）
            if i < response_idx:
                continue
            scores = att_masked[i]  # [seq_len]

            # 1) 按阈值 tau 选成员
            members = torch.nonzero(scores > tau, as_tuple=False).view(-1)

            # 2) 如果没成员，但你希望至少连一点，可以 top-k 兜底
            if USE_TOPK_FALLBACK and members.numel() == 0:
                # 只在 scores > 0 的位置选 top-k
                nonzero = torch.nonzero(scores > 0, as_tuple=False).view(-1)
                if nonzero.numel() == 0:
                    continue
                topk = min(TOPK_PER_ROW, nonzero.numel())
                vals = scores[nonzero]
                _, idx_top = torch.topk(vals, topk)
                members = nonzero[idx_top]

            m_len = members.numel()
            if m_len < MIN_MEMBERS_IN_HE:
                continue

            # 3) 限制成员数：先按 scores 的大小排序再截断
            if m_len > MAX_MEMBERS_PER_HE:
                vals = scores[members]
                topk = min(MAX_MEMBERS_PER_HE, m_len)
                _, idx_top = torch.topk(vals, topk)
                members = members[idx_top]
                m_len = members.numel()

            # 4) 可选：把中心 token i 也加进超边成员
            if INCLUDE_CENTER_TOKEN:
                members = torch.cat([members, members.new_tensor([i])])
                members = torch.unique(members, sorted=True)
                m_len = members.numel()

            if m_len < MIN_MEMBERS_IN_HE:
                continue

            # 控制超边总数
            if he_counter >= MAX_HE_PER_SAMPLE:
                break

            he_node_list.append(members)
            he_id_list.append(torch.full((m_len,), he_counter, dtype=torch.long))

            # he_attr：增加一点信息
            att_vals = att[i, members]  # 用原始 attention（不乘 valid_mask）更符合“注意力强度”
            mean_val = float(att_vals.mean().item())
            max_val = float(att_vals.max().item())
            normalized_head = float(head) / max(1.0, num_heads - 1)

            # clamp 到 [0,1]
            mean_val = max(0.0, min(mean_val, 1.0))
            max_val = max(0.0, min(max_val, 1.0))
            normalized_head = max(0.0, min(normalized_head, 1.0))

            # 这里我把 he_attr 维度扩成 3（mean/max/head_id），HyperCHARM 那边会自动适配 hedge_dim
            he_attr_list.append(torch.tensor([mean_val, max_val, normalized_head], dtype=torch.float32))

            # he_mark：是否是 prompt→response 的跨段超边
            has_prompt_member = (prompt_mask[members]).any()
            if has_prompt_member and i >= response_idx:
                he_mark_list.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
            else:
                he_mark_list.append(torch.tensor([0.0, 1.0], dtype=torch.float32))

            he_member_counts_list.append(m_len)
            he_counter += 1

        if he_counter >= MAX_HE_PER_SAMPLE:
            break
        break

    # assemble tensors
    if len(he_node_list) > 0:
        all_nodes = torch.cat(he_node_list)     # [总 incidence]
        all_hids  = torch.cat(he_id_list)       # [总 incidence]
        he_incidence_index = torch.stack([all_nodes, all_hids], dim=0)  # [2, B]

        he_attr = torch.stack(he_attr_list)         # [E, 3]（原来是 [E, 2]）
        he_mark = torch.stack(he_mark_list)         # [E, 2]
        he_member_counts = torch.tensor(he_member_counts_list, dtype=torch.float32)
    else:
        he_incidence_index = torch.zeros((2, 0), dtype=torch.long)
        he_attr = torch.zeros((0, 3), dtype=torch.float32)  # 对应上面的 3 维
        he_mark = torch.zeros((0, 2), dtype=torch.float32)
        he_member_counts = torch.zeros((0,), dtype=torch.float32)

    obj = {
        "source_id": source_id,
        "response_idx": response_idx,
        "token_ids": token_ids,
        "x": X_V,
        "he_incidence_index": he_incidence_index,
        "he_attr": he_attr,
        "he_mark": he_mark,
        "he_member_counts": he_member_counts,
        "y_token": torch.tensor(hallucination_labels, dtype=torch.float32),
    }

    torch.save(obj, os.path.join(out_dir, f"hypergraph_{fname}"))
    return fname



if __name__ == "__main__":
    # spawn safe multiprocessing
    set_start_method("spawn", force=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".pt")]
    NUM_WORKERS = min(8, os.cpu_count())  # 合理并发，不要过高以免 I/O 抢占

    with Pool(processes=NUM_WORKERS) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_file, files), total=len(files)):
            pass

    print("Preprocessing Done!")
