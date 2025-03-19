import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

# --------------- CGAN 模型定义 ---------------
class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, label_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + label_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = torch.cat([x, labels], dim=1)
        return self.model(x)

# --------------- CGAN 训练函数 ---------------
def train_cgan(real_data, real_labels, num_classes, epochs=1000, batch_size=64, noise_dim=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dim = real_data.shape[1]
    G = Generator(noise_dim, num_classes, data_dim).to(device)
    D = Discriminator(data_dim, num_classes).to(device)
    criterion = nn.BCELoss()  #使用的是BCE loss （可不可以修改？）
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

        if epoch % 100 == 0:
            print(f"[{epoch}/{epochs}] Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    return G


def train_multi_class_cgan(X, y, num_classes, noise_dim=100, epochs=1000, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(noise_dim, num_classes, X.shape[1]).to(device)
    D = Discriminator(X.shape[1], num_classes).to(device)
    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=0.0002)
    opt_D = optim.Adam(D.parameters(), lr=0.0002)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long)
    y_onehot = torch.nn.functional.one_hot(y_tensor, num_classes).float().to(device)

    for epoch in range(epochs):
        idx = torch.randint(0, X.shape[0], (batch_size,))
        real_x = X_tensor[idx]
        real_y = y_onehot[idx]

        z = torch.randn(batch_size, noise_dim).to(device)
        fake_labels = torch.randint(0, num_classes, (batch_size,))
        fake_y = torch.nn.functional.one_hot(fake_labels, num_classes).float().to(device)
        fake_x = G(z, fake_y)

        # Discriminator loss
        real_validity = D(real_x, real_y)
        fake_validity = D(fake_x.detach(), fake_y)
        loss_D = criterion(real_validity, torch.ones_like(real_validity)) + \
                 criterion(fake_validity, torch.zeros_like(fake_validity))

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Generator loss
        gen_validity = D(fake_x, fake_y)
        loss_G = criterion(gen_validity, torch.ones_like(gen_validity))

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if epoch % 100 == 0:
            print(f"[{epoch}] Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")

    return G  # 返回生成器和标签编码器


# --------------- 生成样本 ---------------
def generate_samples(G, target_class, num_classes, n_samples=500, noise_dim=100):
    G.eval()
    device = next(G.parameters()).device
    z = torch.randn(n_samples, noise_dim).to(device)
    labels = torch.full((n_samples,), target_class, dtype=torch.long)
    class_one_hot = torch.nn.functional.one_hot(labels, num_classes).float().to(device)
    samples = G(z, class_one_hot).detach().cpu().numpy()
    return samples

def generate_class_samples(G, le, class_name, num_classes, n_samples=500, noise_dim=100):
    class_idx = le.transform([class_name])[0]
    z = torch.randn(n_samples, noise_dim).to(next(G.parameters()).device)
    labels = torch.full((n_samples,), class_idx, dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(labels, num_classes).float().to(next(G.parameters()).device)
    samples = G(z, one_hot).detach().cpu().numpy()
    return samples

# --------------- 主函数：稀有类增强 ---------------
def cgan_augment(df, rare_classes, generate_per_class=500):
    all_generated = []

    for rare in rare_classes:
        print(f"\n🔍 正在增强类别: {rare}")
        class_df = df[df["Label"] == rare].copy()
        if len(class_df) < 5:
            print(f"⚠️ 样本太少，跳过该类")
            continue

        # 标签编码
        class_df["Label_enc"] = 0  # 只有一个类
        X = class_df.drop(columns=["Label", "Label_enc"])
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0).values
        y = class_df["Label_enc"].values
        num_classes = 1

        # 训练 CGAN
        G = train_cgan(X, y, num_classes=num_classes, epochs=500)

        # 生成样本
        synthetic = generate_samples(G, target_class=0, num_classes=1, n_samples=generate_per_class)
        synthetic_df = pd.DataFrame(synthetic, columns=class_df.drop(columns=["Label", "Label_enc"]).columns)
        synthetic_df["Label"] = rare
        all_generated.append(synthetic_df)

        print(f"✅ 生成完成: {rare} → {generate_per_class} 条")

    all_gen_df = pd.concat(all_generated, ignore_index=True)
    df_augmented = pd.concat([df, all_gen_df], ignore_index=True)

    return df_augmented
def multi_class_cgan_augment(df, target_classes, generate_per_class=500, epochs=500):
    """
    使用多类联合 CGAN 增强多个类别，每类生成 generate_per_class 条样本
    """
    print(f"📊 参与增强的类别: {target_classes}")
    
    # 1. 提取目标类数据
    df_subset = df[df["Label"].isin(target_classes)].copy()
    
    # 2. 标签编码
    le = LabelEncoder()
    df_subset["Label_enc"] = le.fit_transform(df_subset["Label"])  # 编码为 0 ~ n-1

    # 3. 特征处理
    X = df_subset.drop(columns=["Label", "Label_enc"])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).values
    y = df_subset["Label_enc"].values
    num_classes = len(le.classes_)

    print(f"✅ 总共样本数: {len(X)} | 类别数: {num_classes}")

    # 4. 训练联合 CGAN
    print("🧠 开始训练多类联合 CGAN...")
    G = train_multi_class_cgan(X, y, num_classes=num_classes, epochs=epochs)

    # 5. 每类生成 generate_per_class 条样本
    generated_dfs = []
    for class_name in le.classes_:
        print(f"🔬 正在生成类别: {class_name} -> {generate_per_class} 条")
        synth_data = generate_class_samples(G, le, class_name, num_classes, n_samples=generate_per_class)
        synth_df = pd.DataFrame(synth_data, columns=df_subset.drop(columns=["Label", "Label_enc"]).columns)
        synth_df["Label"] = class_name
        generated_dfs.append(synth_df)

    # 6. 合并回原始数据
    all_synth_df = pd.concat(generated_dfs, ignore_index=True)
    df_augmented = pd.concat([df, all_synth_df], ignore_index=True)

    print(f"✅ 增强完成，共生成样本: {len(all_synth_df)} 条")
    return df_augmented

