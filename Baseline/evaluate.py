from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

def evaluate_pipeline(df_final, description, rare_classes=None, rare_raw_df=None, show_rare_report=True):
    """
    参数：
        df_final: 当前要训练和评估的整个数据集（可以包含增强样本）
        description: 模型描述
        rare_classes: 稀有类别名列表
        rare_raw_df: 原始 rare 样本 DataFrame，用于分析增强后模型是否识别到了它们
        show_rare_report: 是否打印报告
    返回：
        clf, X_test, y_test, rare_test_result（当前 test 集中表现）, rare_raw_result（原始样本识别情况）
    """

    X = df_final.drop(columns=["Label"])
    y = df_final["Label"]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\n📊 分类报告 ({description}):")
    print(classification_report(y_test, y_pred, digits=4))

    # -------- 当前测试集中的 Rare Attack 评估 --------
    rare_test_results = []
    if rare_classes is not None:
        y_test = y_test.reset_index(drop=True)
        y_pred = pd.Series(y_pred)

        for cls in rare_classes:
            mask = y_test == cls
            total = mask.sum()
            if total == 0:
                continue
            correct = (y_pred[mask] == cls).sum()
            recall = round(correct / total, 4)
            rare_test_results.append({
                "Class": cls,
                "Test Samples": total,
                "Correctly Predicted": correct,
                "Recall": recall
            })

        rare_df = pd.DataFrame(rare_test_results)
        if show_rare_report:
            print(f"\n📌 当前测试集中的 Rare Attack 分类情况：\n{rare_df.to_string(index=False)}")

    return clf, X_test, y_test, rare_test_results

def evaluate_on_original_rare_samples(model, rare_raw_df):
    """
    用于评估模型是否能正确识别原始 rare 攻击样本。
    
    参数：
        model: 已训练好的分类模型（如 RandomForest）
        rare_raw_df: DataFrame，仅包含原始的 rare 攻击样本（未增强）
    
    返回：
        DataFrame，展示每个攻击类别中原始样本数量、识别数量、Recall
    """

    # 确保特征和标签分离，并进行数值处理
    df = rare_raw_df.copy()
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    X = df.drop(columns=["Label"])
    y = rare_raw_df["Label"]

    # 模型预测
    y_pred = model.predict(X)

    results = []
    for label in y.unique():
        mask = y == label
        total = mask.sum()
        correct = (y_pred[mask] == label).sum()
        recall = round(correct / total, 4) if total > 0 else 0.0
        results.append({
            "Class": label,
            "Original Sample Count": total,
            "Detected Correctly": correct,
            "Recall": recall
        })

    return pd.DataFrame(results)
