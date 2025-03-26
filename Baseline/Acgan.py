import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ======= ACGAN Generator =========
class ACGAN_Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, output_dim):
        super().__init__()
        self.label_embed = nn.Embedding(label_dim, label_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        embedded_labels = self.label_embed(labels)
        x = torch.cat([z, embedded_labels], dim=1)
        return self.model(x)

# ======= ACGAN Discriminator =========
class ACGAN_Discriminator(nn.Module):
    def __init__(self, input_dim, label_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2)
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

# ======= ACGAN 训练函数 =========
def train_acgan(X, y, num_classes, noise_dim=100, epochs=500, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    data_dim = X.shape[1]
    G = ACGAN_Generator(noise_dim, num_classes, data_dim).to(device)
    D = ACGAN_Discriminator(data_dim, num_classes).to(device)

    criterion_adv = nn.BCELoss()
    criterion_cls = nn.CrossEntropyLoss()
    opt_G = optim.Adam(G.parameters(), lr=0.0002)
    opt_D = optim.Adam(D.parameters(), lr=0.0002)

    for epoch in range(epochs):
        idx = torch.randint(0, X.shape[0], (batch_size,))
        real_x = X[idx]
        real_y = y[idx]

        # --------- Train Discriminator ---------
        z = torch.randn(batch_size, noise_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        gen_x = G(z, gen_labels)
        D_real_adv, D_real_cls = D(real_x)
        D_fake_adv, D_fake_cls = D(gen_x.detach())

        loss_D_adv = criterion_adv(D_real_adv, torch.ones_like(D_real_adv)) + \
                     criterion_adv(D_fake_adv, torch.zeros_like(D_fake_adv))
        loss_D_cls = criterion_cls(D_real_cls, real_y)
        loss_D = loss_D_adv + loss_D_cls

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # --------- Train Generator ---------
        gen_x = G(z, gen_labels)
        D_fake_adv, D_fake_cls = D(gen_x)
        loss_G_adv = criterion_adv(D_fake_adv, torch.ones_like(D_fake_adv))
        loss_G_cls = criterion_cls(D_fake_cls, gen_labels)
        loss_G = loss_G_adv + loss_G_cls

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if epoch % 100 == 0:
            print(f"[{epoch}] Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    return G

# ======= 样本生成函数 =======
def generate_acgan_samples(G, class_id, num_classes, n_samples=500, noise_dim=100):
    G.eval()
    device = next(G.parameters()).device
    z = torch.randn(n_samples, noise_dim).to(device)
    labels = torch.full((n_samples,), class_id, dtype=torch.long).to(device)
    samples = G(z, labels).detach().cpu().numpy()
    return samples

# ======= 主函数：增强多个 rare 类别 =======
def acgan_augment(df, rare_classes, generate_per_class=500, epochs=500):
    all_generated = []

    for rare in rare_classes:
        print(f"\n🔧 ACGAN增强中: {rare}")
        class_df = df[df["Label"] == rare].copy()
        if len(class_df) < 5:
            print("⚠️ 样本太少，跳过")
            continue

        class_df["Label_enc"] = 0
        X = class_df.drop(columns=["Label", "Label_enc"])
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0).values
        y = class_df["Label_enc"].values
        num_classes = 1

        G = train_acgan(X, y, num_classes, epochs=epochs)
        synthetic = generate_acgan_samples(G, class_id=0, num_classes=1, n_samples=generate_per_class)

        synthetic_df = pd.DataFrame(synthetic, columns=class_df.drop(columns=["Label", "Label_enc"]).columns)
        synthetic_df["Label"] = rare
        all_generated.append(synthetic_df)

        print(f"✅ 完成: {rare} → +{generate_per_class} 条样本")

    gen_df = pd.concat(all_generated, ignore_index=True)
    df_augmented = pd.concat([df, gen_df], ignore_index=True)
    return df_augmented
