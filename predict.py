# -*- coding: utf-8 -*-
"""预测程序：加载 knn_model.npz，在 test.csv 上评估 4 种 KNN 配置，
输出与实验一致的 Accuracy / Macro F1 分组柱状图（test_comparison.png）。
打包：pyinstaller --onefile --name knn_predict predict.py
使用：把 knn_predict.exe、knn_model.npz、test.csv 放同一目录，双击运行。"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import StandardScalerTrainOnly, KNNClassifier, classification_metrics

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLASSES = ["Star", "Galaxy", "QSO"]
CONFIGS = [
    ("KNN(最优)", "std", "euclidean", "uniform"),
    ("KNN(距离加权)", "std", "euclidean", "distance"),
    ("KNN(曼哈顿)", "std", "manhattan", "uniform"),
    ("KNN(未标准化)", "raw", "euclidean", "uniform"),
]

if getattr(sys, "frozen", False):
    BASE = os.path.dirname(sys.executable)
    # 按控制台真实代码页适配输出编码，避免中文乱码
    try:
        import ctypes
        cp = ctypes.windll.kernel32.GetConsoleOutputCP()
        enc = "utf-8" if cp == 65001 else "gbk"
        sys.stdout.reconfigure(encoding=enc, errors="replace")
        sys.stderr.reconfigure(encoding=enc, errors="replace")
    except Exception:
        pass
else:
    BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    model_path = os.path.join(BASE, "knn_model.npz")
    test_path = os.path.join(BASE, "test.csv")
    if not os.path.exists(test_path):
        test_path = os.path.join(BASE, "data", "splits", "test.csv")
    for p in (model_path, test_path):
        if not os.path.exists(p):
            print("缺少文件:", p)
            return

    print("LAMOST 光谱三分类 KNN 预测")
    print("-" * 40)

    m = np.load(model_path, allow_pickle=True)
    X_tr, y_tr = m["X_train"], m["y_train"]
    mean, std, k = m["mean"], m["std"], int(m["k"])
    features = [str(f) for f in m["features"]]

    df = pd.read_csv(test_path, sep="|")
    X = df[features].to_numpy(np.float32)
    y = df["label"].to_numpy()
    print(f"模型 K={k}，测试集 {len(y)} 条，正在评估 4 种 KNN 配置...")

    # 标准化（与训练流程一致：只用训练集统计量）
    X_tr_s = ((X_tr - mean) / std).astype(np.float32)
    X_s = ((X - mean) / std).astype(np.float32)

    # 预计算三种（标准化×距离）组合的近邻表，K/投票扫描复用
    knns = {}
    for std_name, metric in (("std", "euclidean"), ("std", "manhattan"),
                             ("raw", "euclidean")):
        Xt = X_tr_s if std_name == "std" else X_tr
        Xq = X_s if std_name == "std" else X
        knn = KNNClassifier(k=k, metric=metric,
                            weights="uniform").fit(Xt, y_tr)
        d, i = knn.kneighbors(np.ascontiguousarray(Xq))
        knns[(std_name, metric)] = (knn, d, i)

    labels, accs, f1s = [], [], []
    for name, std_name, metric, weights in CONFIGS:
        knn, d, i = knns[(std_name, metric)]
        knn.k, knn.weights = k, weights
        m = classification_metrics(y, knn.predict_from_neighbors(d, i),
                                   CLASSES)
        labels.append(name)
        accs.append(m["accuracy"])
        f1s.append(m["macro_f1"])
        print(f"  {name:<10} Accuracy={m['accuracy']:.4f}  "
              f"Macro F1={m['macro_f1']:.4f}")

    # 分组柱状图（与实验报告中的对比图一致）
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.6), 4.5))
    ax.bar(x - 0.2, accs, 0.4, label="Accuracy")
    ax.bar(x + 0.2, f1s, 0.4, label="Macro F1")
    for xi, v in zip(x - 0.2, accs):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    for xi, v in zip(x + 0.2, f1s):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_title("各模型测试集表现（同一测试集，仅评估一次）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_png = os.path.join(BASE, "test_comparison.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print("\n完成，结果图片：test_comparison.png")
    if getattr(sys, "frozen", False):
        os.startfile(out_png)  # 自动打开柱状图


if __name__ == "__main__":
    main()
    if getattr(sys, "frozen", False):
        try:
            input("\n按回车退出...")
        except EOFError:
            pass
