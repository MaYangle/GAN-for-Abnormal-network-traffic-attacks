from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def evaluate_pipeline(df_train, description, rare_classes=None, model_cls=None, val_set=None):
    """
    通用模型评估函数，适配任何 sklearn 模型。
    rare_classes：可以传入你关注的攻击类，会逐类输出 recall 分析
    val_set：如果提供，则用外部验证集；否则会自动切分 df_train
    返回：model, X_test, y_test, rare_results_df, le
    """
    df_train = df_train.copy()
    df_train["Label"] = df_train["Label"].astype(str)

    print(f"\n🟩 [Eval] 开始评估: {description}，训练样本数 = {len(df_train)}")

    X = df_train.drop(columns=["Label"])
    y = df_train["Label"]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = model_cls()

    if val_set is not None:
        print(f"🧪 [Eval] 使用外部验证集，样本数 = {len(val_set)}")
        val_set = val_set.copy()
        val_set["Label"] = val_set["Label"].astype(str)
        X_test = val_set.drop(columns=["Label"])
        y_test = val_set["Label"]
        X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
        y_test_encoded = le.transform(y_test)
    else:
        print(f"📊 [Eval] 使用内部划分验证集 (20%)")
        X_train, X_test, y_train, y_test_encoded = train_test_split(
            X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)

    print(f"🔍 [Eval] 模型训练中...")
    model.fit(X, y_encoded)

    print(f"✅ 训练完成，开始预测")
    y_pred = model.predict(X_test)

    print(f"\n📊 分类报告 ({description}):")
    target_names = le.inverse_transform(sorted(np.unique(y_test_encoded)))
    print(classification_report(y_test_encoded, y_pred, target_names=target_names, digits=4))

    # rare 类分析
    rare_results = []
    if rare_classes:
        print(f"\n📌 [Eval] Rare 类别详细评估：")
        for cls_name in rare_classes:
            if cls_name not in le.classes_:
                continue
            cls_id = np.where(le.classes_ == cls_name)[0][0]
            mask = (y_test_encoded == cls_id)
            total = mask.sum()
            if total == 0:
                continue
            correct = (y_pred[mask] == cls_id).sum()
            recall = round(correct / total, 4)
            pred_total = (y_pred == cls_id).sum()
            precision = round(correct / pred_total, 4) if pred_total > 0 else 0
            f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0
            rare_results.append({
                "Class": cls_name,
                "Test Samples": total,
                "Correctly Predicted": correct,
                "Recall": recall,
                "Precision": precision,
                "F1-Score": f1
            })

        rare_df = pd.DataFrame(rare_results)
        print("\n📋 Rare 类别结果：")
        print(rare_df.to_string(index=False))
    else:
        rare_df = None

    print(f"✅ [Eval] 完成评估: {description}")
    return model, X_test, y_test_encoded, rare_df, le
def acgan_augment(df, rare_classes, gen_config, feature_names=None):
    all_generated = []

    for label in rare_classes:
        class_df = df[df["Label"] == label].copy()
        if len(class_df) < 5:
            continue

        class_df["Label_enc"] = 0  # 单类问题时用0
        X = class_df.drop(columns=["Label", "Label_enc"])
        feature_names = X.columns.tolist() if feature_names is None else feature_names
        X = X[feature_names].apply(pd.to_numeric, errors='coerce').fillna(0).values
        y = class_df["Label_enc"].values

        G = train_acgan(X, y, num_classes=1)
        samples = generate_samples(G, target_class=0, num_classes=1, n_samples=gen_config[label])

        syn_df = pd.DataFrame(samples, columns=feature_names)
        syn_df["Label"] = label
        all_generated.append(syn_df)

    gen_df = pd.concat(all_generated, ignore_index=True)
    final_df = pd.concat([df, gen_df], ignore_index=True)
    return final_df, gen_df
