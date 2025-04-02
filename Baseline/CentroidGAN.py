import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =========================================
# ✅ 强化 Generator：更深、更稳定、更 expressive
# =========================================
class StrongGenerator(nn.Module):
    def __init__(self, noise_dim, label_dim, output_dim):
        super().__init__()
        input_dim = noise_dim + label_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LayerNorm(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048), nn.Dropout(0.3), nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024), nn.LayerNorm(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        return self.model(x)

# =========================================
# ✅ 强化 Discriminator：深层特征提取 + 判别输出
# =========================================
class StrongDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, x):
        f = self.features(x)
        out = self.classifier(f)
        return out, f

# =========================================
# ✅ 多项 loss + 分类器反馈 loss（任务驱动型 loss）
# =========================================
def diversity_loss(samples):
    dists = torch.cdist(samples, samples, p=2)
    return -dists.mean()

def task_loss_soft(fake_x, clf, target_label):
    try:
        probs = clf.predict_proba(fake_x.detach().cpu().numpy())[:, target_label]
    except:
        return torch.tensor(0.0).to(fake_x.device)
    return -torch.tensor(probs.mean(), dtype=torch.float32).to(fake_x.device)

# =========================================
# ✅ 训练主函数（强化版）
# =========================================
def train_centroid_gan(X, y, num_classes, real_centroid,
                       clf=None, target_label=0,
                       noise_dim=100, epochs=2000, batch_size=64,
                       feature_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    real_centroid = torch.tensor(real_centroid, dtype=torch.float32).to(device)

    G = StrongGenerator(noise_dim, num_classes, X.shape[1]).to(device)
    D = StrongDiscriminator(X.shape[1]).to(device)

    opt_G = optim.Adam(G.parameters(), lr=0.0002)
    opt_D = optim.Adam(D.parameters(), lr=0.0002)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        # === 1. Sample Real
        idx = torch.randint(0, X.size(0), (batch_size,))
        real_x = X[idx]

        # === 2. Generate Fake
        z = torch.randn(batch_size, noise_dim).to(device)
        labels = torch.zeros(batch_size, dtype=torch.long).to(device)  # single class
        one_hot = torch.nn.functional.one_hot(labels, num_classes).float().to(device)
        fake_x = G(z, one_hot)

        # === 3. Train Discriminator
        D_real, _ = D(real_x)
        D_fake, _ = D(fake_x.detach())
        loss_D = criterion(D_real, torch.ones_like(D_real)) + \
                 criterion(D_fake, torch.zeros_like(D_fake))
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # === 4. Train Generator
        D_fake, _ = D(fake_x)
        loss_adv = criterion(D_fake, torch.ones_like(D_fake))
        loss_cent = torch.norm(fake_x.mean(dim=0) - real_centroid, p=2)
        loss_div = diversity_loss(fake_x)

        # === 5. Task Loss（分类器输出作为惩罚项）
        if clf:
            try:
                probs = clf.predict_proba(fake_x.detach().cpu().numpy())[:, target_label]
                recall_proxy = probs.mean()
                loss_task = -torch.tensor(recall_proxy, dtype=torch.float32).to(device)
            except:
                loss_task = torch.tensor(0.0).to(device)
        else:
            loss_task = torch.tensor(0.0).to(device)

        # === 6. Dynamic Weighting（softmax）
        with torch.no_grad():
            loss_vec = torch.tensor([
                loss_adv.item(),
                loss_cent.item(),
                loss_div.item(),
                loss_task.item()
            ])
            weights = torch.softmax(loss_vec, dim=0).to(device)

        loss_G = (
            weights[0] * loss_adv +
            weights[1] * loss_cent +
            weights[2] * loss_div +
            weights[3] * loss_task
        )

        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        # === 7. Logging
        if epoch % 100 == 0:
            print(f"[{epoch:04d}] G: {loss_G.item():.4f} | "
                  f"Adv: {loss_adv.item():.4f} | "
                  f"C: {loss_cent.item():.4f} | "
                  f"Dv: {loss_div.item():.4f} | "
                  f"Tk: {loss_task.item():.4f} | "
                  f"w: {['%.2f'%w for w in weights.tolist()]}"
            )

    return G

# =========================================
# ✅ 外部增强接口（适配原始主函数）
# =========================================
def centroid_gan_augment(df, rare_classes, gen_config,
                         clf=None,
                         noise_dim=100,
                         feature_names=None,
                         epochs=2000):
    all_generated = []

    for label in rare_classes:
        print(f"\n🚀 CentroidGAN 增强中: {label}")
        class_df = df[df["Label"] == label].copy()
        if len(class_df) < 5:
            print("⚠️ 样本太少，跳过")
            continue

        class_df["Label_enc"] = 0
        X = class_df.drop(columns=["Label", "Label_enc"])
        feature_names = X.columns.tolist() if feature_names is None else feature_names
        X = X[feature_names].apply(pd.to_numeric, errors='coerce').fillna(0).values
        y = class_df["Label_enc"].values
        real_centroid = X.mean(axis=0)

        # ✅ 调用训练函数（自动使用动态权重，无需再传 lambda_xxx）
        G = train_centroid_gan(
            X, y, num_classes=1, real_centroid=real_centroid,
            clf=clf, target_label=0,
            noise_dim=noise_dim,
            epochs=epochs,
            feature_names=feature_names
        )

        # ✅ 生成样本
        samples = generate_samples(G, target_class=0, num_classes=1, n_samples=gen_config[label])
        syn_df = pd.DataFrame(samples, columns=feature_names)
        syn_df["Label"] = label
        all_generated.append(syn_df)

    gen_df = pd.concat(all_generated, ignore_index=True)
    final_df = pd.concat([df, gen_df], ignore_index=True)
    return final_df, gen_df


def generate_samples(G, target_class, num_classes, n_samples=500, noise_dim=100):
    G.eval()
    device = next(G.parameters()).device
    z = torch.randn(n_samples, noise_dim).to(device)
    labels = torch.full((n_samples,), target_class, dtype=torch.long).to(device)
    one_hot = torch.nn.functional.one_hot(labels, num_classes).float().to(device)
    with torch.no_grad():
        return G(z, one_hot).cpu().numpy()
