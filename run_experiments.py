# -*- coding: utf-8 -*-
"""
第7次课作业 —— 实验主流程
  1. 用 sklearn 在小子样本上校验自实现 KNN 的正确性
  2. 标准化（仅用训练集统计量）vs 不标准化
  3. 验证集上选择 K（候选 1,3,5,7,9,15,21），主依据 Macro F1
  4. 距离度量（欧氏/曼哈顿）与投票方式（均匀/距离加权）对照
  5. 测试集只用于最终一次评估
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import StandardScalerTrainOnly, KNNClassifier, classification_metrics

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(BASE, "data", "splits")
RESULT_DIR = os.path.join(BASE, "results")
FIG_DIR = os.path.join(BASE, "figures")
for d in (RESULT_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

SEED = 42
K_CANDIDATES = [1, 3, 5, 7, 9, 15, 21]
KMAX = max(K_CANDIDATES)
CLASSES = ["Star", "Galaxy", "QSO"]
FEATURES = [
    "u_g", "g_r", "r_i", "i_z",
    "c56", "c67", "c14", "c17",
    "log_snru", "log_snrg", "log_snrr", "log_snri", "log_snrz",
    "z", "z_err", "gaia_g_mean_mag",
]


def load_split(name):
    df = pd.read_csv(os.path.join(SPLIT_DIR, f"{name}.csv"), sep="|")
    return df


def xy(df):
    return df[FEATURES].to_numpy(np.float32), df["label"].to_numpy()


# ---------------------------------------------------------- sklearn 校验
def sanity_check(X_tr, y_tr, X_va, y_va):
    from sklearn.neighbors import KNeighborsClassifier
    print("== 自实现 vs sklearn 正确性校验（训练集前3000样本）==")
    sub = slice(0, 3000)
    for metric in ("euclidean", "manhattan"):
        mine = KNNClassifier(k=5, metric=metric, weights="uniform")
        mine.fit(X_tr[sub], y_tr[sub])
        sk = KNeighborsClassifier(n_neighbors=5, metric=metric,
                                  weights="uniform")
        sk.fit(X_tr[sub], y_tr[sub])
        agree = (mine.predict(X_va) == sk.predict(X_va)).mean()
        print(f"  KNN({metric}) 预测一致率: {agree:.4f}")


# ---------------------------------------------------------- 图表
def plot_k_curve(rows, best_k):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = [r["k"] for r in rows]
    ax.plot(ks, [r["val_macro_f1"] for r in rows], "o-", label="Macro F1")
    ax.plot(ks, [r["val_accuracy"] for r in rows], "s--", label="Accuracy")
    ax.axvline(best_k, color="r", ls=":", alpha=0.7, label=f"最优 K={best_k}")
    ax.set_xlabel("K（近邻个数）")
    ax.set_ylabel("验证集指标")
    ax.set_title("验证集指标随 K 的变化（标准化 + 欧氏距离 + 均匀投票）")
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "k_curve.png"), dpi=200)
    plt.close(fig)


def plot_confusion(cm, title, fname):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, mat, name in ((axes[0], cm, "数量"),
                          (axes[1], cm_norm, "归一化（按真实类行）")):
        im = ax.imshow(mat, cmap="Blues", vmin=0)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                v = mat[i, j]
                txt = f"{int(v)}" if name == "数量" else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if v > mat.max() * 0.6 else "black")
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别", labelpad=16)
        ax.set_title(f"混淆矩阵（{name}）")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.subplots_adjust(left=0.09, right=0.965, top=0.86, bottom=0.11,
                        wspace=0.62)
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    plt.close(fig)


def plot_bar_comparison(rows, fname, title):
    labels = [r["name"] for r in rows]
    acc = [r["test_accuracy"] for r in rows]
    f1 = [r["test_macro_f1"] for r in rows]
    x = np.arange(len(labels))
    if fname == "test_comparison.png":
        rows = [r for r in rows if "NB" not in r["name"] and "贝叶斯" not in r["name"]]
        labels = [r["name"] for r in rows]
        acc = [r["test_accuracy"] for r in rows]
        f1 = [r["test_macro_f1"] for r in rows]
        x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.6), 4.5))
    ax.bar(x - 0.2, acc, 0.4, label="Accuracy")
    ax.bar(x + 0.2, f1, 0.4, label="Macro F1")
    for xi, v in zip(x - 0.2, acc):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    for xi, v in zip(x + 0.2, f1):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------- 主流程
def main():
    rng = np.random.default_rng(SEED)
    train_df, val_df, test_df = load_split("train"), load_split("val"), load_split("test")
    X_tr, y_tr = xy(train_df)
    X_va, y_va = xy(val_df)
    X_te, y_te = xy(test_df)
    print(f"train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")

    sanity_check(X_tr, y_tr, X_va[:500], y_va[:500])

    # ---- 标准化：仅用训练集统计量，再变换 val/test（防数据泄漏）----
    scaler = StandardScalerTrainOnly()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    # ---- 预计算近邻（每个 配置 × 数据集 只算一次，K 扫描复用）----
    def neighbors(X_fit, X_q, metric):
        knn = KNNClassifier(k=KMAX, metric=metric, weights="uniform")
        knn.fit(X_fit, y_tr)
        return knn, knn.kneighbors(np.ascontiguousarray(X_q))

    nb_cfg = {}
    for std_name, Xf, Xq_va, Xq_te in (
        ("std", X_tr_s, X_va_s, X_te_s),
        ("raw", X_tr, X_va, X_te),
    ):
        for metric in ("euclidean", "manhattan"):
            knn, (d_va, i_va) = neighbors(Xf, Xq_va, metric)
            _, (d_te, i_te) = neighbors(Xf, Xq_te, metric)
            nb_cfg[(std_name, metric)] = (knn, d_va, i_va, d_te, i_te)
            print(f"近邻计算完成: {std_name}/{metric}")

    def eval_config(std_name, metric, k, weights, on="val"):
        knn, d_va, i_va, d_te, i_te = nb_cfg[(std_name, metric)]
        knn.k, knn.weights = k, weights
        D, I = (d_va, i_va) if on == "val" else (d_te, i_te)
        y = y_va if on == "val" else y_te
        pred = knn.predict_from_neighbors(D, I)
        m = classification_metrics(y, pred, CLASSES)
        m.update({"std": std_name, "metric": metric, "k": k,
                  "weights": weights, "on": on})
        return m

    # ---- 1) 验证集选择 K（标准化 + 欧氏 + 均匀）----
    print("\n== 验证集选择 K ==")
    k_rows = []
    for k in K_CANDIDATES:
        m = eval_config("std", "euclidean", k, "uniform", "val")
        k_rows.append({"k": k, "val_accuracy": m["accuracy"],
                       "val_macro_f1": m["macro_f1"]})
        print(f"K={k:>2}  acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}")
    pd.DataFrame(k_rows).to_csv(os.path.join(RESULT_DIR, "k_selection.csv"),
                                index=False, encoding="utf-8-sig")
    best_k = max(k_rows, key=lambda r: r["val_macro_f1"])["k"]
    stable = [r["k"] for r in k_rows
              if r["val_macro_f1"] >= max(x["val_macro_f1"] for x in k_rows) - 0.005]
    print(f"最优 K={best_k}，±0.005 内的稳定区间 K∈{stable}")
    plot_k_curve(k_rows, best_k)

    # ---- 2) 对照实验（验证集上比较，测试集只在最后统一评估一次）----
    print("\n== 对照实验（验证集）==")
    val_compare = []
    for std_name in ("std", "raw"):
        for metric in ("euclidean", "manhattan"):
            for weights in ("uniform", "distance"):
                m = eval_config(std_name, metric, best_k, weights, "val")
                val_compare.append(m)
                print(f"{std_name}/{metric}/{weights}: "
                      f"acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")

    # ---- 3) 最终测试集评估（仅一次）----
    print("\n== 测试集最终评估 ==")
    final = {}
    configs = [
        ("KNN(最优)", lambda: eval_config("std", "euclidean", best_k,
                                          "uniform", "test")),
        ("KNN(距离加权)", lambda: eval_config("std", "euclidean", best_k,
                                            "distance", "test")),
        ("KNN(曼哈顿)", lambda: eval_config("std", "manhattan", best_k,
                                           "uniform", "test")),
        ("KNN(未标准化)", lambda: eval_config("raw", "euclidean", best_k,
                                             "uniform", "test")),
    ]
    test_rows = []
    for name, fn in configs:
        m = fn()
        final[name] = m
        test_rows.append({"name": name, "test_accuracy": m["accuracy"],
                          "test_macro_f1": m["macro_f1"]})
        print(f"{name}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")

    pd.DataFrame(test_rows).to_csv(os.path.join(RESULT_DIR, "test_comparison.csv"),
                                   index=False, encoding="utf-8-sig")
    plot_bar_comparison(test_rows, "test_comparison.png",
                        "各模型测试集表现（同一测试集，仅评估一次）")

    # ---- 4) 混淆矩阵（KNN 最优）----
    plot_confusion(final["KNN(最优)"]["confusion_matrix"],
                   f"KNN（K={best_k}，标准化+欧氏+均匀投票）测试集混淆矩阵",
                   "confusion_knn.png")

    # ---- 5) 汇总所有指标到 JSON/CSV ----
    def flat(m):
        row = {k: m[k] for k in ("accuracy", "macro_f1", "std", "metric",
                                 "k", "weights", "on")}
        for c in CLASSES:
            row[f"{c}_precision"] = m["per_class"][c]["precision"]
            row[f"{c}_recall"] = m["per_class"][c]["recall"]
            row[f"{c}_f1"] = m["per_class"][c]["f1"]
        return row

    pd.DataFrame([flat(m) for m in val_compare]).to_csv(
        os.path.join(RESULT_DIR, "val_all_configs.csv"),
        index=False, encoding="utf-8-sig")

    summary = {
        "seed": SEED, "best_k": best_k, "stable_k_range": stable,
        "k_curve": k_rows,
        "final_test": {
            name: {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                   "per_class": m["per_class"],
                   "confusion_matrix": m["confusion_matrix"].tolist()}
            for name, m in final.items()
        },
    }
    with open(os.path.join(RESULT_DIR, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    np.savez(os.path.join(RESULT_DIR, "test_neighbors.npz"),
             d_std_e=nb_cfg[("std", "euclidean")][3],
             i_std_e=nb_cfg[("std", "euclidean")][4],
             d_raw_e=nb_cfg[("raw", "euclidean")][3],
             i_raw_e=nb_cfg[("raw", "euclidean")][4],
             y_te=y_te, best_k=best_k)

    print("\n全部结果已保存到 results/ 与 figures/")
    return final, best_k, nb_cfg, scaler


if __name__ == "__main__":
    main()
