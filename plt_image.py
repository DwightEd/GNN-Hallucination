import torch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# 读取图
graphs = torch.load("attributed_graphs/graph_tau0.05_saved_train_layer24_with_QA_num839.pt")
   # 可视化第一个图
image_i=0
for g in graphs:
    if sum(g["hallucination_labels"])>1:
        token_ids = g["token_ids"]
        hallucination_labels = np.asarray(g["hallucination_labels"], dtype=int)

        # 如果形状不是一维，这里强制展平
        hallucination_labels = hallucination_labels.reshape(-1)

        X_V = g["X_V"]
        X_V_reduced = X_V[:, 32:]#只取hidden_state
        E = g["E"]
        X_E = g["X_E"]
        
        # ============================
        # 图结构可视化（基于节点向量相似度的聚类布局）
        # ============================
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
        from scipy.spatial import ConvexHull
        
        # 假设 hallucination_labels 和 X_V 已经准备好
        # hallucination_labels: np.array, 0 = normal, 1 = hallucination
        # X_V: np.array, 节点向量
        
        # t-SNE 降到 2D
        X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_V_reduced)
        
        # KMeans 聚类
        num_clusters = 2  # 可以根据需要调整
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_tsne)
        
        plt.figure(figsize=(10,8))
        
        # 绘制不同簇的凸包
        for cluster in range(num_clusters):
            cluster_idx = np.where(cluster_labels == cluster)[0]
            if len(cluster_idx) < 3:  # 至少三个点才能画凸包
                continue
            points = X_tsne[cluster_idx]
            hull = ConvexHull(points)
            plt.fill(points[hull.vertices,0], points[hull.vertices,1], 
                     edgecolor='black', fill=False, linewidth=1.5, alpha=0.3)
        
        # 绘制节点，颜色和形状区分 hallucination
        for i, (x, y) in enumerate(X_tsne):
            if hallucination_labels[i] == 1:
                plt.scatter(x, y, color='red', marker='D', s=80, label='Hallucination' if i==0 else "")
            else:
                plt.scatter(x, y, color='skyblue', marker='o', s=60, label='Normal' if i==0 else "")
        
        plt.title("t-SNE Node Embedding Visualization with Clusters")
        plt.legend()
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.tight_layout()
        plt.savefig(f"images/only_hidden_state/Node_tSNE_clusters_{g['source_id']}.png", dpi=300)
        plt.close()

        image_i+=1
    if image_i==10:
        break

