import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

# 读取图
graphs = torch.load("attributed_graphs_fixed/pyg_graph_tau0.05_att_saved_train_layer24_with_QA_num839.pt")
image_i=0
for g in graphs:
    # 假设你有 hallucination_labels 信息
    if "y_token" in g and sum(g["y_token"])>1:
        token_ids = g["token_ids"].numpy() if isinstance(g["token_ids"], torch.Tensor) else g["token_ids"]
        hallucination_labels = np.asarray(g["y_token"], dtype=int).reshape(-1)

        X_V = g["x"]  # node features
        X_E = g["edge_attr"]  # edge features
        edge_index = g["edge_index"]  # shape (2, E)
        E = edge_index.T.tolist()  # 转成 list[(i,j)]

        # ============================
        # 1. Node L2 Norm 可视化
        # ============================
        node_l2 = np.linalg.norm(X_V, axis=1)
        plt.figure(figsize=(14,4))
        normal_idx = np.where(hallucination_labels==0)[0]
        hall_idx = np.where(hallucination_labels==1)[0]
        plt.scatter(normal_idx, node_l2[normal_idx], color="skyblue", label="Normal Token (0)")
        plt.scatter(hall_idx, node_l2[hall_idx], color="red", label="Hallucination Token (1)")
        plt.plot(node_l2, alpha=0.3)
        plt.title("Node L2 Norms (blue=normal, red=hallucination)")
        plt.xlabel("Node Index")
        plt.ylabel("L2 Norm")
        plt.legend()
        plt.savefig("images/Node_L2_new.png", dpi=300)
        plt.close()

        # ============================
        # 2. Edge L2 Norm 可视化
        # ============================
        edge_l2 = np.linalg.norm(X_E, axis=1)
        plt.figure(figsize=(14,4))
        edge_colors = ["red" if hallucination_labels[i]==1 else "skyblue" for i,j in E]
        plt.scatter(range(len(edge_l2)), edge_l2, c=edge_colors)
        plt.plot(edge_l2, alpha=0.3)
        plt.title("Edge L2 Norms (blue=normal src, red=hallucination src)")
        plt.xlabel("Edge Index")
        plt.ylabel("L2 Norm")
        plt.savefig("images/Edge_L2_new.png", dpi=300)
        plt.close()

        # ============================
        # 3. t-SNE 可视化 + KMeans 聚类 + 簇凸包
        # ============================
        X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_V)
        num_clusters = 2
        cluster_labels = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(X_tsne)

        plt.figure(figsize=(10,8))
        for cluster in range(num_clusters):
            cluster_idx = np.where(cluster_labels==cluster)[0]
            if len(cluster_idx)<3:
                continue
            points = X_tsne[cluster_idx]
            hull = ConvexHull(points)
            plt.fill(points[hull.vertices,0], points[hull.vertices,1], 
                     edgecolor='black', fill=False, linewidth=1.5, alpha=0.3)

        for i,(x,y) in enumerate(X_tsne):
            if hallucination_labels[i]==1:
                plt.scatter(x, y, color='red', marker='D', s=80, label='Hallucination' if i==0 else "")
            else:
                plt.scatter(x, y, color='skyblue', marker='o', s=60, label='Normal' if i==0 else "")

        plt.title("t-SNE Node Embedding Visualization with Clusters")
        plt.legend()
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.tight_layout()
        plt.savefig(f"images/only_attention_score/Node_tSNE_clusters_new_{g['source_id']}.png", dpi=300)
        plt.close()

        image_i+=1
    if image_i==10:
        break
