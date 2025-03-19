import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_tsne_distribution(benign_df, rare_raw, rare_single, label="RareAttack", sample_size=200):
    """
    输入三个 DataFrame：benign、真实 rare、GAN 增强 rare，进行 t-SNE 可视化。
    """
    feature_cols = [col for col in rare_raw.columns if col != "Label"]

    # 统一采样数量
    real_sample = rare_raw.sample(n=min(sample_size, len(rare_raw)), random_state=42)
    fake_sample = rare_single.sample(n=min(sample_size, len(rare_single)), random_state=42)
    benign_sample = benign_df.sample(n=min(sample_size, len(benign_df)), random_state=42)

    # 拼接数据
    X_all = pd.concat([
        real_sample[feature_cols],
        fake_sample[feature_cols],
        benign_sample[feature_cols]
    ], ignore_index=True)

    labels = (
        ["Real"] * len(real_sample) +
        ["GAN"] * len(fake_sample) +
        ["Benign"] * len(benign_sample)
    )

    # 转换为 t-SNE 空间
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_all)

    # 可视化
    df_vis = pd.DataFrame(X_tsne, columns=["Dim1", "Dim2"])
    df_vis["Type"] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_vis, x="Dim1", y="Dim2", hue="Type", palette="Set1", alpha=0.7, s=60)
    plt.title(f"t-SNE Visualization of {label} - Real vs GAN vs Benign")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
