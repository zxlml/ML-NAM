<div align="center">

# ML-NAM：最大似然神经可加模型

[![Journal](https://img.shields.io/badge/Journal-Information%20Sciences-blue)](https://doi.org/10.1016/j.ins.2026.123104)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ins.2026.123104-9cf)](https://doi.org/10.1016/j.ins.2026.123104)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-orange.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

[English](README.md) | 简体中文

</div>

论文 **"Maximum Likelihood Neural Additive Models"**（发表于 *Information Sciences*，Vol. 736，Article 123104，2026 年）的官方 PyTorch 实现。

## 📢 新闻

* **[2026-09]** 发布 **PyTorch** 重构实现：忠实复现基于核密度估计的最大似然（EML）损失、仿真数据生成，以及论文仿真实验（Table 3 / Table 4）的完整复现流程。
* **[2026-01]** 论文正式发表于 *Information Sciences*。

## 🧠 简介

**神经可加模型（NAM）** 是一种可解释深度学习模型，将预测值建模为若干分量函数之和，每个分量函数由绑定单一输入特征的一维神经网络逼近：

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\hat{f}(x)=\sum_{j=1}^{p}\hat{f}_j(x_j)" />
</p>

然而，现有 NAM 通常局限于**均方误差（MSE）**准则，当数据包含**非高斯噪声**（离群点、重尾噪声）时性能会显著退化。**ML-NAM** 采用最大似然估计进行误差建模，构建**噪声分布感知的可加模型**：通过对残差进行**核密度估计（KDE）**，无需对噪声分布做任何显式假设，即可灵活适应多样的噪声环境。

### ✨ 亮点

* 🎯 **天然鲁棒** —— 在非高斯噪声下，ML-NAM 相比 NAM 将 MSE 降低 **14%–29%**。
* 🧩 **无分布假设** —— 使用固定带宽的核密度估计器建模残差密度，对离群点和重尾分布具有鲁棒性。
* 📐 **理论保证** —— 在温和条件下建立了超额风险的非渐近界，当目标函数属于 Besov 空间时具有多项式衰减的极小极大收敛速率。
* 🔍 **可解释** —— 每个特征的贡献是一个可学习的 1 维形状函数，可视化直观。
* ➕ **ML-NAM+** —— 通过特征选择正则化扩展，用于特征重要性识别。

## 🏗️ 方法

ML-NAM 的核心是基于 mini-batch 残差核密度估计的最大似然（EML）损失（论文式 10–11），实现于 [`graph_builder.py`](code/ML-NAM/model/graph_builder.py)：

```math
K_\sigma(u,u')=\exp\left(-\frac{\|u-u'\|^2}{2\sigma^2}\right),\qquad
\hat h(u)=\frac{1}{n}\sum_i K_\sigma\bigl(y_i-f(x_i),\,u\bigr),\qquad
L_n(f)=\frac{1}{n}\sum_i-\log\hat h\bigl(y_i-f(x_i)\bigr)
```

* 使用**固定带宽 σ**（网格 `{0.01, 0.05, 0.1, 0.5}`），依照论文注记 1 在每个 mini-batch 内计算，避免 O(n²) 开销。
* 实践中采用 **leave-one-out（留一）KDE**：若包含自身项，当残差两两距离超过约 `2σ` 后，每个残差都落在 `-log(1/n)` 的零梯度平台上，且似然会被退化的 δ 密度最大化。留一是极大似然密度估计的标准形式，可使损失适定。
* 目标值按训练集统计 min–max 缩放至 `[0, 1]`，使残差与带宽网格处于同一尺度；模型偏置初始化为目标均值，使初始残差足够致密。
* 三种训练变体共享同一网络结构：
  * **NAM** —— `--loss_type mse`（最小二乘基线，式 2）
  * **ML-NAM** —— `--loss_type eml`（KDE 最大似然损失）
  * **ML-NAM+** —— `--loss_type eml --output_regularization λ`（特征选择）

## 📁 项目结构

```
ML-NAM/
├── code/ML-NAM/
│   ├── model/
│   │   ├── models.py          # NAM / DNN（PyTorch，按特征向量化并行）
│   │   ├── graph_builder.py   # EML（KDE）损失、正则化、评价指标、学习率调度
│   │   ├── data_utils.py      # 仿真数据生成（论文 Sec. 4.1）
│   │   ├── MLNAM_train.py     # 训练循环、早停、形状函数评估
│   │   ├── MLNAM_main.py      # 命令行入口
│   │   └── run_simulation.py  # Table 3 完整复现脚本
│   ├── dataset/               # 数据集（CaliforniaHousing）
│   ├── logs/                  # 实验日志（仅本地保存，不上传）
│   ├── requirements.txt
│   └── run.sh
└── MLNAM.pdf                  # 论文手稿
```

## 🔧 安装

```bash
git clone https://github.com/zxlml/ML-NAM.git
cd ML-NAM/code/ML-NAM

# 推荐 Python >= 3.10
pip install -r requirements.txt
```

依赖：`torch>=2.0`、`numpy>=1.24`、`scikit-learn>=1.0`、`pandas>=1.3`。

## ⚡ 快速开始

以下命令均在 `code/ML-NAM/model/` 目录下运行。

### 1. 在仿真数据集上训练

数据集命名格式为 `sim:<噪声>:<离群比例>:<维度p>:<样本数n>`，其中噪声 ∈ {`gaussian`, `mixture`, `studentt`}，离群比例 ∈ {0, 0.1}，维度 p ∈ {10, 100, 200, 400}：

```bash
# ML-NAM（KDE 最大似然损失）+ 形状函数 MAE（Table 4）
python MLNAM_main.py --dataset sim:gaussian:0:10:1280 --loss_type eml --shape_mae

# NAM 基线（MSE 损失）
python MLNAM_main.py --dataset sim:gaussian:0:10:1280 --loss_type mse

# ML-NAM+（特征选择正则化）
python MLNAM_main.py --dataset sim:gaussian:0:10:1280 --loss_type eml --output_regularization 1e-3

# 重尾噪声，p = 100，10% 离群点
python MLNAM_main.py --dataset sim:studentt:0.1:100:1280 --loss_type eml
```

### 2. 复现完整仿真实验（Table 3）

```bash
python run_simulation.py --repeats 5 --dims 10,100,200,400

# 快速验证（仅 p = 10，重复 2 次）
python run_simulation.py --quick
```

脚本会对每个（噪声 × 离群比例 × 维度）组合训练 **NAM / ML-NAM / ML-NAM+**，报告平均 MSE / MAE（± 标准差）并与论文参考值对比；原始结果保存至本地 `logs/` 目录（实验结果不上传至仓库）。

### 3. 仿真数据生成（论文 Sec. 4.1）

前 10 个特征上的真实可加函数为：

```math
f^{*}(x)=x_1^{3}+x_2+x_3^{1/3}+|x_4|+\sin(x_5)+\cos(x_6^{2})+e^{x_7}+e^{x_8^{2}}+\log(1+x_9^{2})+\log(1+x_{10}^{1/2})
```

* 协变量服从 `U([0,1]^p)`，第 10 维之后的特征不含信息。
* 噪声 ∈ {`N(0,1)`、`0.6N(0,1)+0.4N(0,3)`、`t(3)`}。
* 按比例 r ∈ {0%, 10%} 对标签注入额外的 `N(100, 100)` 离群项。
* 共生成 `n = 1280` 个样本，按 8:1:1 划分 → **1024 训练 / 128 验证 / 128 测试**，与论文一致。

## ⚙️ 超参数

遵循论文的参数选择网格：

| 超参数 | 网格 | 默认值 |
| --- | --- | :-: |
| 损失类型 | {`eml`, `mse`} | `eml` |
| 核带宽 σ | {0.01, 0.05, 0.1, 0.5} | 0.05 |
| 子网络隐藏层数 | {1, 2, 3} | 1 |
| 每层隐藏单元数 | {16, 32, 64} | 32 |
| Dropout / 特征 Dropout | {0, …, 0.5} | 0.1 / 0.1 |
| 批大小 | {32, 64, 128} | 128 |
| 学习率 | 1e-3，多项式衰减（幂 0.5） | 1e-3 |
| 输出正则化 λ（ML-NAM+） | {1e-4, …, 1e-1} | 1e-3 |
| 早停耐心值 | — | 30 轮 |

## ☑️ 待办事项

ML-NAM 仍在持续演进！以下是后续计划：

* [ ] 超参数交叉验证（对 σ、单元数、层数、dropout 进行网格搜索）
* [ ] 真实世界基准数据集（CaliforniaHousing、SCMT、KEGG 等）
* [ ] 已学习分量函数的形状可视化工具
* [ ] 面向大 p、大 n 的 GPU 加速工具

## 🔗 引用

如果本工作对您有帮助，请引用：

```bibtex
@article{chen2026mlnam,
  title   = {Maximum likelihood neural additive models},
  author  = {Chen, Jingyi and Zhang, Xuelin and Yuan, Peipei and Lan, Rushi and Chen, Hong},
  journal = {Information Sciences},
  volume  = {736},
  pages   = {123104},
  year    = {2026},
  issn    = {0020-0255},
  doi     = {10.1016/j.ins.2026.123104}
}
```

## 🙏 致谢

本项目建立在 [Neural Additive Models](https://arxiv.org/abs/2004.13946)（Agarwal et al., 2021）的思想及[官方 NAM 实现](https://github.com/Argonne-NEXT/NAMU)的基础上，感谢开源社区的贡献。

<p align="right"><a href="#ml-nam最大似然神经可加模型">⬆️ 返回顶部</a></p>
