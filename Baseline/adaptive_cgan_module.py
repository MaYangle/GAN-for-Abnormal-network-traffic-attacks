import pandas as pd
import numpy as np
from cgan_module import train_cgan, generate_samples
from evaluate import evaluate_pipeline
from sklearn.ensemble import RandomForestClassifier

def adaptive_cgan_multi_round(
    df,
    rare_classes,
    generate_per_round=1000,
    max_rounds=3,
    recall_threshold=0.95,
    epochs=500,
    model_func=None
):
    """
    多轮自适应增强：每轮训练 + 评估 recall，根据 recall 决定继续增强哪些类。
    返回增强后的 df（原始数据 + 多轮增强）。
    """
    print("\n📌 [Adaptive CGAN - 多轮] 启动自适应增强流程...")

    if model_func is None:
        model_func = lambda: RandomForestClassifier(n_estimators=100, random_state=42)

    df_aug = df.copy()  # 初始化为原始数据
    for round_id in range(1, max_rounds + 1):
        print(f"\n🔁 Round {round_id} / {max_rounds} --------------------------")

        # 构造当前增强轮的训练集（只包含 BENIGN 和稀有类）
        benign_df = df_aug[df_aug["Label"] == "BENIGN"]
        rare_df = df_aug[df_aug["Label"].isin(rare_classes)]
        base_df = pd.concat([benign_df, rare_df], ignore_index=True)

        # 训练模型，评估 rare 类 recall
        clf, _, _, rare_stats, _ = evaluate_pipeline(
            base_df, f"[Adaptive-R{round_id}] Baseline",
            rare_classes=rare_classes, model_cls=model_func
        )

        recalls = {d["Class"]: d["Recall"] for d in rare_stats}
        need_more = [cls for cls in rare_classes if recalls.get(cls, 0) < recall_threshold]

        if not need_more:
            print("🎯 所有 rare 类别 recall 已达标，提前终止增强。")
            break

        # 根据 recall 分配本轮增强数量（越低增强越多）
        weights = {cls: max(1e-3, 1 - recalls.get(cls, 0)) for cls in need_more}
        total_weight = sum(weights.values())
        alloc = {
            cls: int((w / total_weight) * generate_per_round) for cls, w in weights.items()
        }

        print("\n📊 增强分配（基于当前 recall）:")
        for cls in need_more:
            r = recalls.get(cls, 0)
            n = alloc.get(cls, 0)
            print(f"{cls:<30} recall={r:.4f} → 增强 +{n} 条")

        # 本轮增强生成
        all_generated = []
        for cls in need_more:
            class_df = df_aug[df_aug["Label"] == cls]
            if len(class_df) < 5:
                print(f"⚠️ 样本太少，跳过类别: {cls}")
                continue

            gen_num = alloc.get(cls, 0)
            if gen_num <= 0:
                continue

            class_df = class_df.copy()
            class_df["Label_enc"] = 0
            X = class_df.drop(columns=["Label", "Label_enc"])
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0).values
            y = class_df["Label_enc"].values

            G = train_cgan(X, y, num_classes=1, epochs=epochs)
            synthetic = generate_samples(G, target_class=0, num_classes=1, n_samples=gen_num)

            syn_df = pd.DataFrame(
                synthetic, columns=class_df.drop(columns=["Label", "Label_enc"]).columns
            )
            syn_df["Label"] = cls
            all_generated.append(syn_df)
            print(f"✅ {cls} 增强完成 +{gen_num}")

        if all_generated:
            syn_all = pd.concat(all_generated, ignore_index=True)
            df_aug = pd.concat([df_aug, syn_all], ignore_index=True)
        else:
            print("⚠️ 本轮未生成任何数据。")

    print("\n🏁 多轮自适应增强完成。")
    return df_aug
