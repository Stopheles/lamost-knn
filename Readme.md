# LAMOST 光谱三分类 —— KNN 实验

用从零实现的 K 近邻（KNN）完成 LAMOST 光谱 Star / Galaxy / QSO 三分类（第7次课课后实践作业）。

- **数据**：LAMOST DR7 LRS，16 个特征（4 颜色、4 组合颜色、5 个 log 信噪比、z、z_err、Gaia G 星等）；STAR 下采样至 20,000，共 26,162 条
- **划分**：分层抽样 train / val / test = 70% / 15% / 15%，随机种子 42；标准化只用训练集统计量（防泄漏）；测试集仅评估一次
- **实现**：模型部分不依赖 sklearn，纯 numpy 实现；sklearn 仅用于正确性校验

## 文件说明

| 文件 | 作用 |
|---|---|
| `models.py` | numpy 从零实现的 KNN（暴力近邻、欧氏/曼哈顿距离、均匀/距离加权投票）、训练集标准化器、分类指标 |
| `prep_data.py` | 数据准备：生成统一三分类划分 `data/splits/`（train/val/test），只需运行一次 |
| `run_experiments.py` | 实验主流程：sklearn 正确性校验 → 标准化 → 验证集选 K → 距离/投票/标准化对照 → 测试集最终评估，输出 `results/` 与 `figures/` |
| `error_analysis.py` | 错误样本分析：错分方向、低/高信噪比错误率、代表性错误样本、得票校准 |
| `export_model.py` | 导出 KNN 主配置模型到 `knn_model.npz`（训练集特征/标签 + 标准化参数，K=5） |
| `predict.py` | 独立预测程序：加载 `knn_model.npz`，在 `test.csv` 上评估 4 种 KNN 配置并生成对比柱状图；可用 PyInstaller 打包为 exe |

## 安装与运行

```powershell
pip install -r requirements.txt

python prep_data.py          # 生成数据划分（只需一次）
python run_experiments.py    # 完整实验
python error_analysis.py     # 错误分析
```

预测程序打包为 exe（可选）：

```powershell
python export_model.py
python -m PyInstaller --onefile --name knn_predict predict.py
# 把 knn_predict.exe、knn_model.npz、test.csv 放同一目录，双击即可
```

## 主要结果（测试集，K=5）

| 配置 | Accuracy | Macro F1 |
|---|---|---|
| KNN 主配置（标准化+欧氏+均匀） | 0.894 | 0.796 |
| KNN 距离加权 | 0.896 | 0.798 |
| KNN 曼哈顿距离 | 0.907 | 0.815 |
| KNN 未标准化 | 0.960 | 0.897 |

## 结论要点

1. 验证集最优 K=5（Macro F1 峰值），稳定区间 [5, 7]；
2. 本数据上**不标准化反而更好**：最有信息量的特征（z、星等、SNR）恰好方差最大，原始尺度等价于加权，等方差标准化稀释了主要信号；
3. 主要错分在 Galaxy↔Star（测光颜色简并）与低红移 QSO，低信噪比半区错误率约为高信噪比的 4 倍；
4. 近邻得票比例是未校准的粗粒度置信参考，不能当作可信概率。
