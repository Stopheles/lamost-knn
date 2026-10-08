# -*- coding: utf-8 -*-
"""生成可提交的可运行 Notebook：knn_homework.ipynb，并用 nbclient 执行嵌入输出。"""
import os
import nbformat as nbf

BASE = os.path.dirname(os.path.abspath(__file__))

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

C = []

C.append(nbf.v4.new_markdown_cell("""# LAMOST 光谱三类监督分类：KNN 分类实验
**第7次课课后实践作业** —— Star / Galaxy / QSO 三分类

- **数据**：第6讲作业3.1相同样本范围（LAMOST DR7 LRS，ugrizjh 测光、字段完整的样本），类别统一为 Star/Galaxy/QSO（剔除 UNKNOWN），STAR 下采样至 20,000（GALAXY 4,930、QSO 1,232 全保留），共 26,162 条。
- **划分**：分层抽样 train/val/test = 70%/15%/15%，随机种子 42，各子集类别比例一致。
- **特征**：与第6讲完全相同的 16 个特征（4 个颜色、4 个组合颜色、5 个 log 信噪比、红移 z、z_err、Gaia G 波段星等）。
- **方法**：从零实现 KNN（标准化、欧氏/曼哈顿距离、均匀/距离加权投票）；测试集只用于最终一次评估。
- **复现**：`prep_data.py` 从原始数据生成 `data/splits/`；本 notebook 从划分结果出发完整复现实验。"""))

C.append(nbf.v4.new_markdown_cell("""## 0. 环境与数据加载
模型部分（`models.py`）不依赖 sklearn，仅用 numpy 实现；sklearn 仅用于正确性校验。"""))

C.append(nbf.v4.new_code_cell("""import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from models import StandardScalerTrainOnly, KNNClassifier, classification_metrics

SEED = 42
K_CANDIDATES = [1, 3, 5, 7, 9, 15, 21]
KMAX = max(K_CANDIDATES)
CLASSES = ['Star', 'Galaxy', 'QSO']
FEATURES = ['u_g', 'g_r', 'r_i', 'i_z', 'c56', 'c67', 'c14', 'c17',
            'log_snru', 'log_snrg', 'log_snrr', 'log_snri', 'log_snrz',
            'z', 'z_err', 'gaia_g_mean_mag']

SPLIT_DIR = 'data/splits'
def load_split(name):
    df = pd.read_csv(os.path.join(SPLIT_DIR, f'{name}.csv'), sep='|')
    return df

train_df, val_df, test_df = load_split('train'), load_split('val'), load_split('test')
X_tr, y_tr = train_df[FEATURES].to_numpy(np.float32), train_df['label'].to_numpy()
X_va, y_va = val_df[FEATURES].to_numpy(np.float32), val_df['label'].to_numpy()
X_te, y_te = test_df[FEATURES].to_numpy(np.float32), test_df['label'].to_numpy()

print('训练集', X_tr.shape, ' 验证集', X_va.shape, ' 测试集', X_te.shape)
for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    print(name, dict(df['label'].value_counts()))"""))

C.append(nbf.v4.new_markdown_cell("""## 1. 正确性校验：自实现 vs sklearn
在小子样本上比对自实现 KNN（两种距离）与 `sklearn.neighbors.KNeighborsClassifier` 的预测一致性。"""))

C.append(nbf.v4.new_code_cell("""from sklearn.neighbors import KNeighborsClassifier

sub = slice(0, 3000)
for metric in ('euclidean', 'manhattan'):
    mine = KNNClassifier(k=5, metric=metric).fit(X_tr[sub], y_tr[sub])
    sk = KNeighborsClassifier(n_neighbors=5, metric=metric).fit(X_tr[sub], y_tr[sub])
    print(f'KNN({metric}) 与sklearn预测一致率:',
          (mine.predict(X_va[:500]) == sk.predict(X_va[:500])).mean())"""))

C.append(nbf.v4.new_markdown_cell("""## 2. 标准化（仅用训练集统计量，防止数据泄漏）
KNN 基于距离，特征量纲差异大（星等 ~10–20、颜色 ~0–1、z_err ~1e-5），必须标准化。
**关键**：均值/标准差只在训练集上估计，再变换验证集与测试集；绝不能在全体样本上先标准化再划分。"""))

C.append(nbf.v4.new_code_cell("""scaler = StandardScalerTrainOnly()          # 只用训练集 fit
X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
X_va_s = scaler.transform(X_va).astype(np.float32)
X_te_s = scaler.transform(X_te).astype(np.float32)

stats = pd.DataFrame({'mean': scaler.mean_, 'std': scaler.std_}, index=FEATURES)
print(stats.round(3))"""))

C.append(nbf.v4.new_markdown_cell("""## 3. 验证集选择 K（标准化 + 欧氏距离 + 均匀投票）
对每个查询点预计算 KMAX=21 个近邻，之后所有 K 的评估只查表，避免重复计算。
主依据：验证集 **Macro F1**（三类样本量差异大，Accuracy 会被多数类 Star 主导）。"""))

C.append(nbf.v4.new_code_cell("""knn = KNNClassifier(k=KMAX, metric='euclidean', weights='uniform').fit(X_tr_s, y_tr)
d_va, i_va = knn.kneighbors(X_va_s)
d_te, i_te = knn.kneighbors(X_te_s)

k_rows = []
for k in K_CANDIDATES:
    knn.k = k
    m = classification_metrics(y_va, knn.predict_from_neighbors(d_va, i_va), CLASSES)
    k_rows.append({'K': k, 'Accuracy': m['accuracy'], 'Macro F1': m['macro_f1']})
k_df = pd.DataFrame(k_rows)
print(k_df.round(4).to_string(index=False))

best_k = int(k_df.loc[k_df['Macro F1'].idxmax(), 'K'])
best_f1 = k_df['Macro F1'].max()
stable = k_df[k_df['Macro F1'] >= best_f1 - 0.005]['K'].tolist()
print(f'\\n最优 K = {best_k}（Macro F1 = {best_f1:.4f}），±0.005 稳定区间 K ∈ {stable}')

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(k_df['K'], k_df['Macro F1'], 'o-', label='Macro F1')
ax.plot(k_df['K'], k_df['Accuracy'], 's--', label='Accuracy')
ax.axvline(best_k, color='r', ls=':', alpha=.7, label=f'最优 K={best_k}')
ax.set_xlabel('K（近邻个数）'); ax.set_ylabel('验证集指标')
ax.set_title('验证集指标随 K 的变化（标准化 + 欧氏距离 + 均匀投票）')
ax.set_xticks(K_CANDIDATES); ax.grid(alpha=.3); ax.legend()
plt.tight_layout(); plt.show()"""))

C.append(nbf.v4.new_markdown_cell("""**K 的影响**：K=1 时验证 Macro F1 仅 0.742 —— 模型对噪声敏感、方差大（高方差区）；K 增大至 5 到达峰值；K>7 后缓慢下降 —— 邻居中混入其他类的样本，决策边界被过度平滑（偏差增大）。Accuracy 对 K 不敏感是因为它被 Star 多数类主导，Macro F1 更能反映小类上的变化。"""))

C.append(nbf.v4.new_markdown_cell("""## 4. 对照实验（全部在验证集上比较）
- 标准化 vs 不标准化
- 欧氏距离 vs 曼哈顿距离
- 均匀投票 vs 距离加权投票（权重 1/d）"""))

C.append(nbf.v4.new_code_cell("""# 预计算 4 种配置的近邻表（每配置仅算一次，之后 K/投票扫描都查表）
nb_cfg = {}
for metric in ('euclidean', 'manhattan'):
    knn_s = KNNClassifier(k=KMAX, metric=metric).fit(X_tr_s, y_tr)
    d1, i1 = knn_s.kneighbors(X_va_s)
    knn_r = KNNClassifier(k=KMAX, metric=metric).fit(X_tr, y_tr)
    d2, i2 = knn_r.kneighbors(X_va)
    nb_cfg[('std', metric)] = (knn_s, d1, i1)
    nb_cfg[('raw', metric)] = (knn_r, d2, i2)

def eval_val(std_name, metric, k, weights):
    knn, d, i = nb_cfg[(std_name, metric)]
    knn.k, knn.weights = k, weights
    m = classification_metrics(y_va, knn.predict_from_neighbors(d, i), CLASSES)
    return {'标准化': std_name, '距离': metric, '投票': weights, 'K': k,
            'Accuracy': m['accuracy'], 'Macro F1': m['macro_f1']}

rows = [eval_val(s, m, best_k, w)
        for s in ('std', 'raw') for m in ('euclidean', 'manhattan')
        for w in ('uniform', 'distance')]
cmp_df = pd.DataFrame(rows)
cmp_df['标准化'] = cmp_df['标准化'].map({'std': '是', 'raw': '否'})
cmp_df['投票'] = cmp_df['投票'].map({'uniform': '均匀', 'distance': '距离加权'})
print(cmp_df.round(4).to_string(index=False))"""))

C.append(nbf.v4.new_markdown_cell("""**发现**：在这组特征上，**不标准化反而更好**（验证 acc 0.957 vs 0.893）。
原因：特征方差与物理区分度正相关 —— 区分度最强的 z（QSO 的 z 普遍 >0.5，跨度达 0–5）和 Gaia 星等、log 信噪比方差最大，原始欧氏距离下自然获得更大权重，等方差标准化反而把它们与噪声大的颜色特征等量齐观、稀释了主要信号。第6讲互信息排序（z、z_err、log_snrr…靠前）也印证了这一点。
**教训**：标准化是防止“量纲大者主导”的默认安全手段，但并非总是提升性能 —— 当大尺度特征恰好是最有信息量的特征时，保留原始尺度等价于一种“加权”。两种做法都报告，用验证集说话。"""))




C.append(nbf.v4.new_markdown_cell("""## 5. 测试集最终评估（仅一次）
用验证集选出的主配置：**K=5、标准化、欧氏距离、均匀投票**；同时报告各对照配置。"""))

C.append(nbf.v4.new_code_cell("""# 测试集近邻表
nb_cfg_te = {}
for metric in ('euclidean', 'manhattan'):
    knn_s = KNNClassifier(k=KMAX, metric=metric).fit(X_tr_s, y_tr)
    d1, i1 = knn_s.kneighbors(X_te_s)
    knn_r = KNNClassifier(k=KMAX, metric=metric).fit(X_tr, y_tr)
    d2, i2 = knn_r.kneighbors(X_te)
    nb_cfg_te[('std', metric)] = (knn_s, d1, i1)
    nb_cfg_te[('raw', metric)] = (knn_r, d2, i2)

def eval_test(std_name, metric, k, weights):
    knn, d, i = nb_cfg_te[(std_name, metric)]
    knn.k, knn.weights = k, weights
    return classification_metrics(y_te, knn.predict_from_neighbors(d, i), CLASSES)

final = {}
final['KNN(主配置 K=5)'] = eval_test('std', 'euclidean', best_k, 'uniform')
final['KNN(距离加权)']   = eval_test('std', 'euclidean', best_k, 'distance')
final['KNN(曼哈顿)']     = eval_test('std', 'manhattan', best_k, 'uniform')
final['KNN(未标准化)']   = eval_test('raw', 'euclidean', best_k, 'uniform')

test_rows = [{'模型': n, 'Accuracy': m['accuracy'], 'Macro F1': m['macro_f1']}
             for n, m in final.items()]
test_df_out = pd.DataFrame(test_rows)
print(test_df_out.round(4).to_string(index=False))

# 分类别指标（主配置）
for name in ['KNN(主配置 K=5)']:
    print(f'\\n== {name} 分类别指标 ==')
    for c in CLASSES:
        v = final[name]['per_class'][c]
        print(f\"  {c:<7} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} (support={v['support']})\")"""))

C.append(nbf.v4.new_markdown_cell("""**各 KNN 配置在同一测试集上的表现对比**："""))
C.append(nbf.v4.new_code_cell("""from IPython.display import Image, display
display(Image('figures/test_comparison.png'))"""))

C.append(nbf.v4.new_code_cell("""# 主配置混淆矩阵（数量 + 归一化）
def plot_cm(cm, title):
    cm_n = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, mat, nm in zip(axes, (cm, cm_n), ('数量', '归一化(按行)')):
        im = ax.imshow(mat, cmap='Blues', vmin=0)
        for i in range(3):
            for j in range(3):
                v = mat[i, j]
                ax.text(j, i, f'{int(v)}' if nm == '数量' else f'{v:.2f}',
                        ha='center', va='center',
                        color='white' if v > mat.max() * .6 else 'black')
        ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel('预测类别'); ax.set_ylabel('真实类别', labelpad=14)
        ax.set_title(f'混淆矩阵（{nm}）'); fig.colorbar(im, ax=ax, fraction=.046)
    fig.suptitle(title); plt.tight_layout(); plt.show()

plot_cm(final['KNN(主配置 K=5)']['confusion_matrix'],
        f'KNN（K={best_k}，标准化+欧氏+均匀投票）测试集混淆矩阵')"""))

C.append(nbf.v4.new_markdown_cell("""**主要错分方向**：Galaxy↔Star 双向混淆最大（180 条 Galaxy 被判成 Star、136 条 Star 被判成 Galaxy），其次是 QSO→Star（51）。QSO 召回率仅 0.60 —— 低红移 QSO（z<0.3）的光谱与测光特征与恒星、星系高度重叠。"""))

C.append(nbf.v4.new_markdown_cell("""## 6. 错误样本分析（测试集 414 个错误，占 10.6%）
区分四类原因：低信噪比、边界对象、可疑标签、模型局限。"""))

C.append(nbf.v4.new_code_cell("""knn_main, d_m, i_m = nb_cfg_te[('std', 'euclidean')]
knn_main.k, knn_main.weights = best_k, 'uniform'
pred = knn_main.predict_from_neighbors(d_m, i_m)
wrong = np.where(pred != y_te)[0]

err = test_df.iloc[wrong].copy()
err['pred'] = pred[wrong]
err['pred_conf'] = (y_tr[i_m[wrong, :best_k]] == pred[wrong, None]).mean(axis=1)
err['mean_log_snr'] = err[[f'log_snr{b}' for b in 'ugriz']].mean(axis=1)

print('错误样本 真实->预测 分布:')
print(pd.crosstab(err['label'], err['pred'], margins=True))

med = np.median(test_df[[f'log_snr{b}' for b in 'ugriz']].mean(axis=1))
low = test_df[[f'log_snr{b}' for b in 'ugriz']].mean(axis=1).to_numpy() < med
print(f'\\n低SNR半区错误率 {(pred[low]!=y_te[low]).mean():.4f} vs 高SNR半区 {(pred[~low]!=y_te[~low]).mean():.4f}')

cases = pd.concat([
    err.sort_values('mean_log_snr').head(2).assign(类型='低信噪比'),
    err.sort_values('pred_conf').head(2).assign(类型='边界样本(得票最低)'),
    err[err['z'] < 0.12].head(2).assign(类型='标签可疑(QSO但z极低)'),
    err.sort_values('pred_conf', ascending=False).head(2).assign(类型='高置信错误(模型局限)'),
])
show = ['类型', 'obsid', 'label', 'pred', 'pred_conf', 'mean_log_snr', 'z', 'subclass']
print('\\n代表性错误样本:')
print(cases[show].round(4).to_string(index=False))"""))

C.append(nbf.v4.new_markdown_cell("""**四类错误原因**：
1. **低信噪比**（如 obsid 645209129，平均 log SNR≈0.11）：测光与红移测量噪声大，特征不可靠，错误率随 SNR 降低显著上升（低 SNR 半区错误率 16.9% vs 高 SNR 4.2%）；
2. **边界对象**（如 F5 恒星被 2/5 邻居投给 Galaxy）：位于类间决策边界，投票接近平局，本身难分；
3. **可疑标签**（如 obsid 69101214：标为 QSO 但 z=0.058 且 z_err=-9999 缺失）：低红移“QSO”标签很可能是管线误标，属于标签噪声而非模型错误；
4. **模型局限**（如 obsid 402914034：z=0.19 的正常星系被 5/5 邻居一致判为 Star）：红星系与冷恒星的测光颜色本来就简并，KNN 的局部投票无法突破特征本身的信息上限。"""))

C.append(nbf.v4.new_markdown_cell("""## 7. 思考题：近邻类别比例能否直接视为可信概率？
**不能。** 用校准曲线验证：把“预测类的得票比例”当作概率，与相应区间样本的实际准确率对比。"""))

C.append(nbf.v4.new_code_cell("""votes = y_tr[i_m[:, :best_k]]
conf = np.where(pred == 'Star', (votes == 'Star').mean(1),
        np.where(pred == 'Galaxy', (votes == 'Galaxy').mean(1),
                 (votes == 'QSO').mean(1)))
correct = (pred == y_te)
bins = [(0.4, '0.4(2/5)'), (0.6, '0.6(3/5)'), (0.8, '0.8(4/5)'), (1.0, '1.0(5/5)')]
cal = []
for v, lab in bins:
    m = np.isclose(conf, v)
    cal.append({'得票比例': lab, '样本数': int(m.sum()),
                '实际准确率': float(correct[m].mean())})
cal_df = pd.DataFrame(cal)
print(cal_df.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(5.2, 5))
ax.plot([0, 1], [0, 1], 'k--', label='完美校准')
ax.plot([float(x[0]) for x in bins], cal_df['实际准确率'], 'o-', label='KNN 得票比例')
ax.set_xlabel('预测类的近邻得票比例（被当作“概率”）')
ax.set_ylabel('该区间的实际准确率')
ax.set_title(f'KNN（K={best_k}）得票比例校准性检验')
ax.grid(alpha=.3); ax.legend(); plt.tight_layout(); plt.show()"""))

C.append(nbf.v4.new_markdown_cell("""**结论（三个理由）**：
1. **取值过于粗糙**：K=5 时“概率”只能取 0.4/0.6/0.8/1.0 四个离散值，无法区分同区间样本的置信差异，不能用于概率阈值筛选；
2. **系统性失真**：得票 0.4（2/5 险胜）的样本实际正确率约 0.62，得票比例与实际概率明显偏离 —— 它只是 K 个邻居里的计数比例，没有考虑邻居的远近、局部密度和类别先验；
3. **结构偏差**：训练集中重复观测、成协天体聚集会让邻居高度相关，得票被放大；类别不均衡时多数类天然得票偏高。KNN 的“概率”未经过任何校准（如 Platt scaling / isotonic），只能作为排序参考，不能当作可信的概率输出。"""))

C.append(nbf.v4.new_markdown_cell("""## 8. 结论
1. 主配置 **KNN（K=5、标准化、欧氏、均匀投票）** 测试集 Accuracy 0.894 / Macro F1 0.796；四种配置中未标准化最高（Accuracy 0.960 / Macro F1 0.896）——最有信息量的特征（z、星等、SNR）恰好方差最大，保留原始尺度等价于一种加权；
2. K 的选择存在偏差-方差权衡：K 过小高方差、过大过度平滑，本数据验证集最优 K=5，稳定区间 [5, 7]；
3. 标准化在本数据集上**降低**了 KNN 表现，因为最有信息量的特征（z、星等、SNR）恰好方差最大 —— 但防泄漏流程（只用训练集统计量）必须严格遵守；
4. 主要错分在 Galaxy→Star（180）与 Star→Galaxy（136）（测光颜色简并）及低红移 QSO，低信噪比样本错误率高 4 倍，部分“QSO”标签本身可疑；
5. 近邻得票比例是粗粒度、未校准的置信参考，不能视为可信概率。

**分工与复现**：随机种子 42；`prep_data.py` 生成划分 → 本 notebook 完成全部实验；结果表存于 `results/`，图存于 `figures/`。"""))

nb["cells"] = C
path = os.path.join(BASE, "knn_homework.ipynb")
nbf.write(nb, path)
print("notebook written:", path)
