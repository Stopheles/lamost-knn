# -*- coding: utf-8 -*-
"""
第7次课作业 —— 数据准备
样本范围与特征与第6讲作业3.1完全一致（来自 lamost_train.csv / lamost_val.csv），
仅将类别从四类(GALAXY/QSO/STAR/UNKNOWN)统一为三类(Galaxy/QSO/Star)，
UNKNOWN 样本按作业要求剔除。
划分方式：分层抽样 train/val/test = 70%/15%/15%，随机种子 42。
"""
import os
import json
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR = os.path.join(DATA_DIR, "splits")
SEED = 42
STAR_CAP = 20_000          # STAR 样本过多，随机下采样至 2 万；GALAXY/QSO 全保留
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15   # test = 0.15

FEATURES = [
    "u_g", "g_r", "r_i", "i_z",
    "c56", "c67", "c14", "c17",
    "log_snru", "log_snrg", "log_snrr", "log_snri", "log_snrz",
    "z", "z_err", "gaia_g_mean_mag",
]
META_COLS = ["obsid", "ra", "dec", "subclass"]
CLASS_MAP = {"STAR": "Star", "GALAXY": "Galaxy", "QSO": "QSO"}

USECOLS = META_COLS + ["class"] + FEATURES


def load_all():
    frames = []
    for name in ["lamost_train.csv", "lamost_val.csv"]:
        path = os.path.join(DATA_DIR, name)
        print(f"读取 {name} ...")
        df = pd.read_csv(path, sep="|", usecols=lambda c: c.strip() in USECOLS)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print(f"合并后总样本数: {len(df):,}")
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = load_all()

    # ---- 标签统一为 Star / Galaxy / QSO ----
    df["class"] = df["class"].astype(str).str.strip().str.upper()
    n_unknown = int((df["class"] == "UNKNOWN").sum())
    df = df[df["class"].isin(CLASS_MAP.keys())].copy()
    df["label"] = df["class"].map(CLASS_MAP)
    print(f"剔除 UNKNOWN: {n_unknown:,} 条；剩余 {len(df):,} 条")

    # ---- 缺失值检查 ----
    n_missing = int(df[FEATURES].isna().sum().sum())
    missing_per_col = df[FEATURES].isna().sum()
    print(f"特征缺失值总数: {n_missing:,}")
    print(missing_per_col[missing_per_col > 0])

    # ---- 异常值概况（仅报告，不删除；第6讲导出时已做完整性筛选）----
    desc = df[FEATURES].describe().T
    desc["skew"] = df[FEATURES].skew()
    desc.to_csv(os.path.join(OUT_DIR, "feature_stats.csv"), encoding="utf-8-sig")

    # ---- 类别统计 ----
    class_counts = df["label"].value_counts()
    print("\n三类样本量（原始）:")
    print(class_counts)
    print("\n类别比例:")
    print((class_counts / len(df)).round(4))

    # ---- STAR 下采样，控制规模与类别失衡 ----
    star_idx = df.index[df["label"] == "Star"].to_numpy()
    drop_n = len(star_idx) - STAR_CAP
    if drop_n > 0:
        drop = rng.choice(star_idx, size=drop_n, replace=False)
        df = df.drop(index=drop)
    df = df.reset_index(drop=True)
    print(f"\nSTAR 下采样至 {STAR_CAP:,} 后，总样本: {len(df):,}")

    # ---- 分层划分 70/15/15 ----
    train_idx, val_idx, test_idx = [], [], []
    for cls, grp in df.groupby("label"):
        idx = rng.permutation(grp.index.to_numpy())
        n = len(idx)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    for name, idx in splits.items():
        part = df.loc[idx].reset_index(drop=True)
        part.to_csv(os.path.join(OUT_DIR, f"{name}.csv"),
                    sep="|", index=False, encoding="utf-8-sig")
        print(f"\n[{name}] n={len(part):,}")
        print((part["label"].value_counts() / len(part)).round(4))

    # ---- 保存划分索引与环境信息，保证可复现 ----
    manifest = {
        "seed": SEED,
        "source": ["lamost_train.csv", "lamost_val.csv"],
        "sample_rule": "第6讲作业3.1相同样本范围(ugrizjh,完整样本)；剔除UNKNOWN；"
                       f"STAR随机下采样至{STAR_CAP}",
        "split": {"train": TRAIN_RATIO, "val": VAL_RATIO,
                  "test": round(1 - TRAIN_RATIO - VAL_RATIO, 2)},
        "features": FEATURES,
        "class_map": CLASS_MAP,
        "n_unknown_dropped": n_unknown,
        "n_missing_feature_values": n_missing,
        "pandas": pd.__version__, "numpy": np.__version__,
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("\n划分完成，索引与说明已保存到 data/splits/")


if __name__ == "__main__":
    main()
