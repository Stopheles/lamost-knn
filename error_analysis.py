# -*- coding: utf-8 -*-
"""
第7次课作业 —— 错误样本分析 + 思考题（近邻比例能否视为可信概率）
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import KNNClassifier

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(BASE, "data", "splits")
RESULT_DIR = os.path.join(BASE, "results")
FIG_DIR = os.path.join(BASE, "figures")
CLASSES = ["Star", "Galaxy", "QSO"]
FEATURES = [
    "u_g", "g_r", "r_i", "i_z", "c56", "c67", "c14", "c17",
    "log_snru", "log_snrg", "log_snrr", "log_snri", "log_snrz",
    "z", "z_err", "gaia_g_mean_mag",
]


def main():
    test_df = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"), sep="|")
    train_df = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"), sep="|")
    X_te = test_df[FEATURES].to_numpy(np.float32)
    y_te = test_df["label"].to_numpy()

    # 重新拟合标准化器（仅用训练集）
    from models import StandardScalerTrainOnly
    scaler = StandardScalerTrainOnly()
    X_tr_s = scaler.fit_transform(
        train_df[FEATURES].to_numpy(np.float32)).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    npz = np.load(os.path.join(RESULT_DIR, "test_neighbors.npz"))
    D, I, best_k = npz["d_std_e"], npz["i_std_e"], int(npz["best_k"])
    y_tr = train_df["label"].to_numpy()

    knn = KNNClassifier(k=best_k, metric="euclidean", weights="uniform")
    knn.fit(X_tr_s, y_tr)
    pred = knn.predict_from_neighbors(D, I)

    wrong = np.where(pred != y_te)[0]
    print(f"测试集错误样本数: {len(wrong)}/{len(y_te)}")

    # ---- 错误类别统计 ----
    err_df = test_df.iloc[wrong].copy()
    err_df["pred"] = pred[wrong]
    tab = pd.crosstab(err_df["label"], err_df["pred"], margins=True)
    print("\n错误样本的 真实->预测 分布:")
    print(tab)

    # ---- 每个错误样本：邻居投票比例、信噪比、红shift一致性 ----
    votes = y_tr[I[wrong, :best_k]]
    conf = (votes == pred[wrong, None]).mean(axis=1)   # 预测类得票比例
    snr = err_df[["log_snru", "log_snrg", "log_snrr", "log_snri", "log_snrz"]].mean(axis=1)
    z = err_df["z"].to_numpy()

    def z_flag(cls, zz):
        if cls == "Star":
            return "z≈0" if abs(zz) < 0.01 else f"Star但z={zz:.3f}(可疑)"
        if cls == "Galaxy":
            return "z正常" if zz < 0.6 else f"Galaxy但z={zz:.2f}(偏高)"
        return "z正常(QSO)" if zz > 0.3 else f"QSO但z={zz:.3f}(偏低,可疑)"

    err_df["pred_conf"] = conf
    err_df["mean_log_snr"] = snr
    err_df["z_check"] = [z_flag(c, zz) for c, zz in zip(err_df["label"], z)]

    cols = ["obsid", "label", "pred", "pred_conf", "mean_log_snr", "z",
            "z_err", "subclass", "z_check"]
    err_df[cols].to_csv(os.path.join(RESULT_DIR, "test_errors.csv"),
                        index=False, encoding="utf-8-sig")

    # ---- 代表性错误样本 ----
    print("\n== 代表性错误样本 ==")
    cases = {}

    # 1) 低信噪比
    low_snr = err_df.sort_values("mean_log_snr").head(3)
    cases["低信噪比"] = low_snr

    # 2) 边界样本：预测类得票比例最低（最接近平局）
    boundary = err_df.sort_values("pred_conf").head(3)
    cases["边界样本"] = boundary

    # 3) 标签可疑：红移与类别明显不一致
    sus = err_df[err_df["z_check"].str.contains("可疑")].head(3)
    cases["标签可疑"] = sus

    # 4) 高置信错误：模型局限（邻居高度一致但判错）
    high_conf_wrong = err_df.sort_values("pred_conf", ascending=False).head(3)
    cases["高置信错误"] = high_conf_wrong

    rep_rows = []
    for kind, g in cases.items():
        for _, r in g.iterrows():
            rep_rows.append({"类型": kind, **{c: r[c] for c in cols}})
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(os.path.join(RESULT_DIR, "error_cases.csv"),
               index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(rep.round(4).to_string(index=False))

    # ---- 低SNR vs 整体错误率 ----
    med_snr = np.median(test_df[[f"log_snr{b}" for b in "ugriz"]]
                        .mean(axis=1))
    all_snr = test_df[[f"log_snr{b}" for b in "ugriz"]].mean(axis=1).to_numpy()
    low_mask = all_snr < med_snr
    print(f"\n低SNR半区错误率: {(pred[low_mask] != y_te[low_mask]).mean():.4f}"
          f"  高SNR半区错误率: {(pred[~low_mask] != y_te[~low_mask]).mean():.4f}")

    # ---- 思考题：近邻得票比例是否可信概率（校准曲线）----
    votes_all = y_tr[I[:, :best_k]]
    conf_all = np.zeros(len(y_te))
    for ci, c in enumerate(CLASSES):
        conf_all = np.where(pred == c,
                            (votes_all == c).mean(axis=1), conf_all)
    correct = (pred == y_te)

    bins = np.linspace(0, 1.0001, 6)
    labels_b = ["(0.2,0.4]", "(0.4,0.6]", "(0.6,0.8]", "(0.8,1.0]"]
    bidx = np.digitize(conf_all, bins[1:-1])
    cal_rows = []
    for bi in range(4):
        m = bidx == bi
        if m.sum() == 0:
            continue
        cal_rows.append({
            "bin": labels_b[bi], "n": int(m.sum()),
            "mean_vote_share": float(conf_all[m].mean()),
            "empirical_accuracy": float(correct[m].mean()),
        })
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(os.path.join(RESULT_DIR, "calibration.csv"),
               index=False, encoding="utf-8-sig")
    print("\n== 得票比例 vs 实际准确率（校准）==")
    print(cal.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="完美校准")
    ax.plot(cal["mean_vote_share"], cal["empirical_accuracy"], "o-",
            color="tab:blue", label="KNN 近邻得票比例")
    for _, r in cal.iterrows():
        ax.annotate(f"n={r['n']}", (r["mean_vote_share"], r["empirical_accuracy"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_xlabel("预测类的近邻得票比例（被当作“概率”）")
    ax.set_ylabel("该区间样本的实际准确率")
    ax.set_title(f"KNN（K={best_k}）得票比例的校准性检验")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "calibration.png"), dpi=200)
    plt.close(fig)

    # ---- 补充：错误样本的平均特征 vs 正确样本（模型局限讨论用）----
    feat_comp = pd.DataFrame({
        "正确样本均值": test_df.iloc[np.where(correct)[0]][FEATURES].mean(),
        "错误样本均值": test_df.iloc[wrong][FEATURES].mean(),
    })
    feat_comp.to_csv(os.path.join(RESULT_DIR, "error_feature_compare.csv"),
                     encoding="utf-8-sig")
    print("\n错误样本已导出: results/test_errors.csv, results/error_cases.csv")


if __name__ == "__main__":
    main()
