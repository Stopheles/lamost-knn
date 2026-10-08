# -*- coding: utf-8 -*-
"""
第7次课作业 —— 从零实现 KNN（不依赖 sklearn 的模型部分）

KNN:
  - 标准化参数只在训练集上估计（均值/标准差），再应用到验证/测试集，避免数据泄漏；
  - 支持欧氏距离与曼哈顿距离；
  - 支持均匀投票与距离加权投票（权重 1/(d+eps)）；
  - 近邻搜索为暴力实现（numpy 批量矩阵运算），返回每个查询点最近的 kmax 个邻居。
"""
import numpy as np


# ---------------------------------------------------------------- 标准化
class StandardScalerTrainOnly:
    """只用训练集拟合的 z-score 标准化器。"""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ---------------------------------------------------------------- KNN
class KNNClassifier:
    """
    参数
    ----
    k            : 近邻个数
    metric       : 'euclidean' 或 'manhattan'
    weights      : 'uniform' 均匀投票 / 'distance' 距离加权投票
    search       : 'brute'（暴力，批量矩阵实现）
    """

    def __init__(self, k=5, metric="euclidean", weights="uniform",
                 search="brute"):
        assert metric in ("euclidean", "manhattan")
        assert weights in ("uniform", "distance")
        self.k = k
        self.metric = metric
        self.weights = weights
        self.search = search

    def fit(self, X, y):
        self.X_train_ = np.ascontiguousarray(X, dtype=np.float32)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def _pairwise_dist(self, Q):
        """Q (m,d) 与训练集 (n,d) 的成对距离，返回 (m,n) float32。"""
        X = self.X_train_
        if self.metric == "euclidean":
            # ||q-x||^2 = ||q||^2 + ||x||^2 - 2 q.x
            q2 = (Q * Q).sum(axis=1, keepdims=True)
            x2 = (X * X).sum(axis=1, keepdims=True).T
            d2 = q2 + x2 - 2.0 * (Q @ X.T)
            np.maximum(d2, 0.0, out=d2)
            return np.sqrt(d2, out=d2)
        else:  # manhattan
            # 分块累加 |q-x|，避免一次性 (m,n,d) 大数组
            m, n, d = Q.shape[0], X.shape[0], X.shape[1]
            D = np.empty((m, n), dtype=np.float32)
            for j0 in range(0, d, 4):
                j1 = min(j0 + 4, d)
                D += np.abs(Q[:, None, j0:j1] - X[None, :, j0:j1]).sum(axis=2)
            return D

    def kneighbors(self, Q, k=None, batch=512):
        """返回每个查询点最近的 k 个邻居: (dist, idx)，形状 (m,k)。"""
        k = k or self.k
        m = Q.shape[0]
        D_all = np.empty((m, k), dtype=np.float32)
        I_all = np.empty((m, k), dtype=np.int64)
        for s in range(0, m, batch):
            e = min(s + batch, m)
            D = self._pairwise_dist(Q[s:e])
            part = np.argpartition(D, k - 1, axis=1)[:, :k]
            # 按距离排序
            order = np.argsort(
                np.take_along_axis(D, part, axis=1), axis=1)
            idx = np.take_along_axis(part, order, axis=1)
            I_all[s:e] = idx
            D_all[s:e] = np.take_along_axis(D, idx, axis=1)
        return D_all, I_all

    def predict_from_neighbors(self, D, I):
        """给定邻居距离 D (m,k) 与训练集索引 I (m,k)，进行类别投票。"""
        m, k = I.shape
        k_eff = min(self.k, k)
        D, I = D[:, :k_eff], I[:, :k_eff]
        w = (np.ones_like(D) if self.weights == "uniform"
             else 1.0 / (D + 1e-12))
        votes = self.y_train_[I]                      # (m,k)
        out = np.empty(m, dtype=self.classes_.dtype)
        for i in range(m):
            score = {}
            for c, wi in zip(votes[i], w[i]):
                score[c] = score.get(c, 0.0) + wi
            out[i] = max(score, key=score.get)
        return out

    def predict(self, Q):
        D, I = self.kneighbors(np.ascontiguousarray(Q, dtype=np.float32),
                               k=self.k)
        return self.predict_from_neighbors(D, I)

    def predict_proba_from_neighbors(self, D, I):
        """近邻类别比例（思考题用：它能否当作可信概率？）"""
        k_eff = min(self.k, I.shape[1])
        votes = self.y_train_[I[:, :k_eff]]
        proba = np.zeros((I.shape[0], len(self.classes_)))
        for ci, c in enumerate(self.classes_):
            proba[:, ci] = (votes == c).mean(axis=1)
        return proba


# ---------------------------------------------------------------- 指标
def confusion_matrix(y_true, y_pred, classes):
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    pos = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[pos[t], pos[p]] += 1
    return cm


def classification_metrics(y_true, y_pred, classes):
    """返回 accuracy, macro_f1 以及每类 precision/recall/f1。"""
    cm = confusion_matrix(y_true, y_pred, classes)
    per_class = {}
    f1s = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": int(cm[i, :].sum())}
        f1s.append(f1)
    acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s)),
            "per_class": per_class, "confusion_matrix": cm}
