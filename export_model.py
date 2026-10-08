# -*- coding: utf-8 -*-
"""导出 KNN 主配置模型（K=5、标准化、欧氏、均匀投票）到 knn_model.npz。
KNN 的"模型" = 训练集特征 + 标签 + 标准化参数。"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
FEATURES = [
    "u_g", "g_r", "r_i", "i_z", "c56", "c67", "c14", "c17",
    "log_snru", "log_snrg", "log_snrr", "log_snri", "log_snrz",
    "z", "z_err", "gaia_g_mean_mag",
]

df = pd.read_csv(os.path.join(BASE, "data", "splits", "train.csv"), sep="|")
X = df[FEATURES].to_numpy(np.float32)
y = df["label"].to_numpy()

mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1.0

out = os.path.join(BASE, "knn_model.npz")
np.savez_compressed(out, X_train=X, y_train=y, mean=mean, std=std,
                    k=np.int64(5), features=np.array(FEATURES))
print("saved:", out, f"train={X.shape}, k=5")
