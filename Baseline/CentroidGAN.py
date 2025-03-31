# ✅ 改进版 CGAN + ACGAN 模型与训练流程，融合论文 SYN-GAN 的结构与思路

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

# =====================================
# ✅ 通用工具：Boxplot 分布匹配检测函数
# =====================================
def is_boxplot_similar(real_df, fake_df, feature_columns, tolerance=0.15):
    real_desc = real_df[feature_columns].describe()
    fake_desc = fake_df[feature_columns].describe()
    for feat in feature_columns:
        for stat in ["25%", "50%", "75%", "min", "max"]:
            r = real_desc.loc[stat, feat]
            f = fake_desc.loc[stat, feat]
            if abs(r - f) > tolerance * abs(r + 1e-6):
                return False
    return True

# =====================================
# ✅ 改进 Generator（适用于 CGAN/ACGAN）
# =====================================
class BetterGenerator(nn.Module):
    def __init__(self, noise_dim, label_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, output_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        return self.model(x)

# =====================================
# ✅ 改进 Discriminator（用于 CGAN）
# =====================================
class BetterDiscriminator(nn.Module):
    def __init__(self, input_dim, label_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + label_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = torch.cat([x, labels], dim=1)
        return self.model(x)

# =====================================
# ✅ 改进 ACGAN 判别器（增加分类输出）
# =====================================
class ACGANDiscriminator(nn.Module):
    def __init__(self, input_dim, label_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.cls_head = nn.Linear(256, label_dim)

    def forward(self, x):
        features = self.shared(x)
        validity = self.adv_head(features)
        cls_logits = self.cls_head(features)
        return validity, cls_logits

# =====================================
# ✅ 改进 CGAN 训练函数（加入 boxplot early stopping）
# =====================================
def train_cgan(real_data, real_labels, num_classes, epochs=1000, batch_size=64, noise_dim=100, feature_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dim = real_data.shape[1]
    G = BetterGenerator(noise_dim, num_classes, data_dim).to(device)
    D = BetterDiscriminator(data_dim, num_classes).to(device)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=0.0002)
    opt_D = optim.Adam(D.parameters(), lr=0.0002)

    real_data = torch.tensor(real_data, dtype=torch.float32).to(device)
    real_labels_tensor = torch.tensor(real_labels, dtype=torch.long)
    one_hot_labels = torch.nn.functional.one_hot(real_labels_tensor, num_classes).float().to(device)

    for epoch in range(epochs):
        idx = torch.randperm(real_data.size(0))[:batch_size]
        real_batch = real_data[idx]
        real_label_batch = one_hot_labels[idx]

        z = torch.randn(batch_size, noise_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (batch_size,))
        gen_one_hot = torch.nn.functional.one_hot(gen_labels, num_classes).float().to(device)
        fake_data = G(z, gen_one_hot)

        real_validity = D(real_batch, real_label_batch)
        fake_validity = D(fake_data.detach(), gen_one_hot)

        loss_real = criterion(real_validity, torch.ones_like(real_validity))
        loss_fake = criterion(fake_validity, torch.zeros_like(fake_validity))
        loss_D = loss_real + loss_fake

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        gen_validity = D(fake_data, gen_one_hot)
        loss_G = criterion(gen_validity, torch.ones_like(gen_validity))

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if epoch % 100 == 0 and feature_names is not None:
            fake_np = fake_data.detach().cpu().numpy()
            real_np = real_data[:batch_size].detach().cpu().numpy()
            fake_df = pd.DataFrame(fake_np, columns=feature_names)
            real_df = pd.DataFrame(real_np, columns=feature_names)
            if is_boxplot_similar(real_df, fake_df, feature_names):
                print(f"✅ Boxplot matched @ Epoch {epoch}. Early stopping.")
                break

    return G

# =====================================
# ✅ 改进 ACGAN 训练函数（含 boxplot 检查）
# =====================================
def train_acgan_with_centroid_loss(X, y, num_classes, real_centroid, lambda_centroid=2.0,
                                   noise_dim=100, epochs=500, batch_size=64, feature_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    real_centroid = torch.tensor(real_centroid, dtype=torch.float32).to(device)

    data_dim = X.shape[1]
    G = BetterGenerator(noise_dim, num_classes, data_dim).to(device)
    D = ACGANDiscriminator(data_dim, num_classes).to(device)

    criterion_adv = nn.BCELoss()
    criterion_cls = nn.CrossEntropyLoss()
    opt_G = optim.Adam(G.parameters(), lr=0.0002)
    opt_D = optim.Adam(D.parameters(), lr=0.0002)

    for epoch in range(epochs):
        idx = torch.randint(0, X.shape[0], (batch_size,))
        real_x = X[idx]
        real_y = y[idx]

        z = torch.randn(batch_size, noise_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        one_hot = torch.nn.functional.one_hot(gen_labels, num_classes).float().to(device)
        gen_x = G(z, one_hot)

        # 判别器 loss
        D_real_adv, D_real_cls = D(real_x)
        D_fake_adv, D_fake_cls = D(gen_x.detach())

        loss_D_adv = criterion_adv(D_real_adv, torch.ones_like(D_real_adv)) + \
                     criterion_adv(D_fake_adv, torch.zeros_like(D_fake_adv))
        loss_D_cls = criterion_cls(D_real_cls, real_y)
        loss_D = loss_D_adv + loss_D_cls
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # 生成器 loss（含质心 loss）
        gen_x = G(z, one_hot)
        D_fake_adv, D_fake_cls = D(gen_x)
        loss_G_adv = criterion_adv(D_fake_adv, torch.ones_like(D_fake_adv))
        loss_G_cls = criterion_cls(D_fake_cls, gen_labels)
        centroid_fake = gen_x.mean(dim=0)
        loss_centroid = torch.norm(centroid_fake - real_centroid, p=2)
        loss_G = loss_G_adv + loss_G_cls + lambda_centroid * loss_centroid

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if epoch % 100 == 0 and feature_names is not None:
            fake_np = gen_x.detach().cpu().numpy()
            real_np = X[:batch_size].detach().cpu().numpy()
            fake_df = pd.DataFrame(fake_np, columns=feature_names)
            real_df = pd.DataFrame(real_np, columns=feature_names)
            if is_boxplot_similar(real_df, fake_df, feature_names):
                print(f"✅ ACGAN Boxplot matched @ Epoch {epoch}. Early stopping.")
                break

    return G

# =====================================
# ✅ 引入质心计算
# =====================================
def train_cgan_centroid_only(real_data, real_labels, num_classes, real_centroid,
                              lambda_centroid=1.0, epochs=1000, batch_size=64,
                              noise_dim=100, feature_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dim = real_data.shape[1]
    G = BetterGenerator(noise_dim, num_classes, data_dim).to(device)

    opt_G = optim.Adam(G.parameters(), lr=0.0002)

    real_data = torch.tensor(real_data, dtype=torch.float32).to(device)
    real_centroid = torch.tensor(real_centroid, dtype=torch.float32).to(device)
    real_labels_tensor = torch.tensor(real_labels, dtype=torch.long)
    one_hot_labels = torch.nn.functional.one_hot(real_labels_tensor, num_classes).float().to(device)

    for epoch in range(epochs):
        z = torch.randn(batch_size, noise_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (batch_size,))
        gen_one_hot = torch.nn.functional.one_hot(gen_labels, num_classes).float().to(device)
        fake_data = G(z, gen_one_hot)

        centroid_fake = fake_data.mean(dim=0)
        loss_centroid = torch.norm(centroid_fake - real_centroid, p=2)
        loss_G = lambda_centroid * loss_centroid

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if epoch % 100 == 0:
            print(f"[Epoch {epoch}] CentroidOnly Loss: {loss_centroid.item():.4f}")

    return G


# =====================================
# ✅ CGAN 样本生成封装
# =====================================
def generate_samples(G, target_class, num_classes, n_samples=500, noise_dim=100):
    G.eval()
    device = next(G.parameters()).device
    z = torch.randn(n_samples, noise_dim).to(device)
    labels = torch.full((n_samples,), target_class, dtype=torch.long)
    class_one_hot = torch.nn.functional.one_hot(labels, num_classes).float().to(device)
    samples = G(z, class_one_hot).detach().cpu().numpy()
    return samples

# =====================================
# ✅ ACGAN 样本生成封装
# =====================================
def generate_acgan_samples(G, class_id, num_classes, n_samples=500, noise_dim=100):
    G.eval()
    device = next(G.parameters()).device
    z = torch.randn(n_samples, noise_dim).to(device)
    labels = torch.full((n_samples,), class_id, dtype=torch.long).to(device)
    one_hot = torch.nn.functional.one_hot(labels, num_classes).float().to(device)
    samples = G(z, one_hot).detach().cpu().numpy()
    return samples

# =====================================
# ✅ CGAN增强（支持 per-class 控制生成数量）
# =====================================
def cgan_augment_with_centroid_loss(df, rare_classes, gen_config,
                                    lambda_centroid=2.0, feature_names=None):
    all_generated = []
    print("质量心为", lambda_centroid)
    for rare in rare_classes:
        print(f"\n🎯 CGAN增强 + 质心对齐中: {rare}")
        class_df = df[df["Label"] == rare].copy()
        if len(class_df) < 5:
            print("⚠️ 样本太少，跳过")
            continue

        class_df["Label_enc"] = 0
        X = class_df.drop(columns=["Label", "Label_enc"])
        feature_names = X.columns.tolist() if feature_names is None else feature_names
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0).values
        y = class_df["Label_enc"].values

        # 计算 rare 类质心
        real_centroid = X.mean(axis=0)

        # 训练带质心 loss 的 CGAN
        G = train_cgan_centroid_only(
            real_data=X,
            real_labels=y,
            num_classes=1,
            real_centroid=real_centroid,
            lambda_centroid=lambda_centroid,
            epochs=500,
            feature_names=feature_names
        )

        # 生成增强样本
        synthetic = generate_samples(G, target_class=0, num_classes=1, n_samples=gen_config[rare])
        synthetic_df = pd.DataFrame(synthetic, columns=feature_names)
        synthetic_df["Label"] = rare
        all_generated.append(synthetic_df)

    gen_df = pd.concat(all_generated, ignore_index=True)
    return pd.concat([df, gen_df], ignore_index=True), gen_df


# =====================================
# ✅ ACGAN增强（支持 per-class 控制生成数量）
# =====================================
def acgan_augment_with_centroid_loss(df, rare_classes, gen_config,
                                     lambda_centroid=100.0, feature_names=None):
    all_generated = []
    print("质量心为", lambda_centroid)
    for rare in rare_classes:
        print(f"\n🧠 ACGAN增强 + 质心对齐中: {rare}")
        class_df = df[df["Label"] == rare].copy()
        if len(class_df) < 5:
            print("⚠️ 样本太少，跳过")
            continue

        class_df["Label_enc"] = 0
        X = class_df.drop(columns=["Label", "Label_enc"])
        feature_names = X.columns.tolist() if feature_names is None else feature_names
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0).values
        y = class_df["Label_enc"].values

        # 计算 rare 类质心
        real_centroid = X.mean(axis=0)

        # 训练带质心 loss 的 ACGAN
        G = train_acgan_with_centroid_loss(
            X=X,
            y=y,
            num_classes=1,
            real_centroid=real_centroid,
            lambda_centroid=lambda_centroid,
            feature_names=feature_names
        )

        # 生成增强样本
        synthetic = generate_acgan_samples(G, class_id=0, num_classes=1, n_samples=gen_config[rare])
        synthetic_df = pd.DataFrame(synthetic, columns=feature_names)
        synthetic_df["Label"] = rare
        all_generated.append(synthetic_df)

    gen_df = pd.concat(all_generated, ignore_index=True)
    return pd.concat([df, gen_df], ignore_index=True), gen_df
