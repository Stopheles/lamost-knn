# LAMOST 光谱三分类 —— KNN 实验

用从零实现的 KNN 完成 LAMOST 光谱 Star / Galaxy / QSO 三分类（第7次课作业）。
26,162 条样本、16 个特征，分层划分 train/val/test = 70%/15%/15%，种子 42，测试集仅评估一次。

## 文件说明

| 文件 | 作用 |
|---|---|
| `运行实验.bat` | 一键运行：双击即可跑完整实验（数据检查 → 实验 → 错误分析） |
| `models.py` | numpy 从零实现的 KNN、标准化器、分类指标 |
| `prep_data.py` | 生成 `data/splits/` 统一划分（只需跑一次） |
| `run_experiments.py` | 实验主流程：校验、选 K、对照实验、测试集评估 |
| `error_analysis.py` | 错误样本分析 |
| `knn_homework.ipynb` | 可提交的实验 notebook（由 `build_notebook.py` 生成） |
| `make_summary.py` | 生成 `实验摘要.docx` |
| `regen_test_comparison.py` | 单独重绘测试集对比图 |
| `refresh_notebook_image.py` | 刷新 notebook 里嵌入的对比图 |
| `ppt/` | 汇报 PPT |
| `data/`、`results/`、`figures/` | 数据划分、指标表、图 |

## 运行

双击 `运行实验.bat`，或命令行：

```powershell
python prep_data.py && python run_experiments.py && python error_analysis.py && python make_summary.py
```

## 结果（测试集，K=5）

| 配置 | Accuracy | Macro F1 |
|---|---|---|
| 主配置（标准化+欧氏+均匀） | 0.894 | 0.796 |
| 距离加权 | 0.896 | 0.798 |
| 曼哈顿距离 | 0.907 | 0.815 |
| 未标准化 | 0.960 | 0.897 |

要点：最优 K=5；不标准化反而更好（高信息量特征方差最大）；主要错分为 Galaxy↔Star 简并与低红移 QSO；近邻得票比例不能当作可信概率。
