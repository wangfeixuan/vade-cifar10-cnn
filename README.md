# VaDE → CIFAR-10（老师要求的 CNN 迁移实验）

打开 **[VaDE_CIFAR10.ipynb](VaDE_CIFAR10.ipynb)**，选择能导入 PyTorch 的 Python 3 内核，从上到下运行。模型、损失、训练与图表的代码全部在 Notebook 内，无需依赖 tools 目录。


## 已完成的全量训练

50,000 张训练图、10,000 张独立测试图；50 轮 AE 预训练 + 300 轮 VaDE 联合训练已全部完成。使用 Apple MPS、seed=42、Gaussian 观测模型；采用最后一轮模型，没有根据测试指标选择轮次。

| 测试方法 | ACC | NMI | ARI | 占用簇数 |
|---|---:|---:|---:|---:|
| AE+GMM | 23.81% | 0.1153 | 0.0559 | 10 |
| VaDE（第 300 轮） | **25.14%** | 0.1125 | **0.0626** | 10 |

VaDE 的 ACC 比 AE+GMM 提高 1.33 个百分点，ARI 提高，NMI 略降；不能据此声称所有指标都改善。当前为单次随机种子的 CIFAR-10 迁移实验，聚类效果仍有限，不等同于达到论文的 MNIST 精度。

AE 像素 MSE 从 0.03201 降至 0.01837；VaDE 最终总损失为 -421.2683。预训练约 7.9 分钟，联合训练约 48.9 分钟，两阶段合计约 56.8 分钟（不含 GMM、评估和绘图）。

主 Notebook 已保留全部执行输出与图表，11 个代码单元格均完成且没有错误输出；最终权重已核验为第 300 轮、参数均为有限值。

- [全量指标 metrics.csv](results/full_20261117_025603_757775/metrics.csv)
- [完整结果 summary.json](results/full_20261117_025603_757775/summary.json)
- 最终模型及优化器检查点 `vade_last.pt` 保留在本地实验目录；因体积较大，不纳入 Git 仓库
- [训练曲线](results/full_20261117_025603_757775/loss_curves.png)
- [核验记录 validation.json](results/full_20261117_025603_757775/validation.json)

## 已完成的快速试跑（历史对照）

此前快速试跑已使用 Apple MPS 从头执行全部 11 个代码单元格，0 个错误输出，Notebook 内保留 7 幅图（含输入示例）。训练样本 5,000、测试样本 2,000，AE 5 轮 + VaDE 10 轮，seed=42。

| 测试方法 | ACC | NMI | ARI | 占用簇数 |
|---|---:|---:|---:|---:|
| AE+GMM | 22.75% | 0.0835 | 0.0445 | 10 |
| VaDE | 22.55% | 0.0818 | 0.0421 | 10 |

这次短训练没有显示 VaDE 超过 AE+GMM；重建保留部分色彩和轮廓但细节模糊，真实类别在潜空间投影中明显重叠。不能据此声称已复现论文精度，**全量训练已完成，结果见上节**。

实际记录：[metrics.csv](results/quick_20261117_025159_560006/metrics.csv)、[summary.json](results/quick_20261117_025159_560006/summary.json)、[图表及权重目录](results/quick_20261117_025159_560006/)。源代码数学校验 4 项全部通过，另已通过真实 MPS 前向/反向更新检查。

快速试跑的完整输出另存于 [VaDE_CIFAR10_quick.ipynb](VaDE_CIFAR10_quick.ipynb)。

## 内容

- CIFAR-10 RGB 输入，3×32×32；三层卷积 encoder、三层反卷积 decoder。
- 卷积 AE 预训练、对角 GMM 初始化、VaDE 端到端联合训练。
- AE+GMM 与 VaDE 对照；ACC（匈牙利匹配）、NMI、ARI、占用簇数。
- 损失曲线、原图/重建、按簇排列的测试图、簇与类别计数、PCA 潜空间、按高斯分量生成。
- 固定随机种子、保存实际数据索引、配置、环境、权重、优化器状态、预测及指标。

## 运行

```bash
cd /Users/wangfeixuan/Desktop/科研/VADE
python3 -m pip install -r requirements.txt
```

然后用 Jupyter 或 VS Code 打开 Notebook。也可执行：

```bash
python3 tools/run_notebook.py
```

`MODE="quick"` 的配置（Notebook 中通过 `os.environ.get` 设置）：5,000 张训练图、2,000 张测试图，5 轮 AE + 10 轮 VaDE。当前主 Notebook 已默认使用 `MODE="full"`： 50,000 张训练图 / 10,000 张测试图、50 轮 AE + 300 轮 VaDE。终端也可用：

```bash
VADE_MODE=full python3 tools/run_notebook.py
```

`smoke` 只用 256/128 张及 1+1 轮，适合检查环境。不同模式的结果不可混写。每次运行输出到 `results/<模式>_<时间>/`，不会覆盖旧实验。重新运行训练格会继续修改内存中的模型；要从头复现实验，请重启内核并 Run All。

执行脚本显式使用当前可用的 `python3` 内核。

## 方法与原仓库的关系

原仓库是 [GuHY777/VaDE-pytorch](https://github.com/GuHY777/VaDE-pytorch)，实际使用 MLP。本实现依据其模型、预训练、GMM 初始化与论文公式编写自包含 Notebook，并按教师要求改为 CNN。逐项差异和源文件 blob SHA 已记录在 Notebook。

连续 RGB 默认采用固定观测方差 0.1 的 Gaussian 负对数似然；这是迁移实验的简化。若要对照原仓库的重建目标，将 `CFG` 中 `likelihood` 改为 `"bce"` 后完整重跑。对 [0,1] 连续像素使用 BCE 是软像素重建目标。没有对图像做 [-1,1] 标准化。

训练沿用原仓库的采样责任度近似。修正混合权重的概率约束和 CUDA 硬编码，并将预测改为基于编码均值的确定性规则。官方训练集用于拟合；官方测试集仅用于评估，不根据测试结果选择轮次。这与原仓库合并训练/测试样本的协议不同，不能直接对比其 MNIST ACC。

Gaussian NLL 包含归一化常数，连续概率密度可以大于 1，因此重建 NLL 或总损失为负不一定表示错误。观察趋势、有限性和重建效果。

## 数据与依赖

数据默认下载到本目录 `data/`。下载失败时可将官方 `cifar-10-python.tar.gz` 放入该目录再运行。官方压缩包 MD5 为 `c58f30108f718f92721af3b95e74349a`；代码仍校验证书及数据完整性。macOS Python 的代理失效时自动尝试 curl 官方 HTTPS 下载。也可用 `VADE_DATA_DIR` 指定已有数据目录。

`requirements.txt` 提供跨平台依赖名称；每次执行保存实际 Python、PyTorch、torchvision 与 NumPy 版本。CPU、Apple MPS、CUDA 自动选择，不使用外部预训练权重。

## 校验

```bash
python3 tools/verify_notebook.py
```

检查直接读取 Notebook 的实现，验证 GMM 概率与 sklearn 一致、损失与 `torch.distributions` 的独立 KL/NLL 计算一致、梯度可用，以及 ACC 的编号置换/簇塌缩案例。

`tools/build_notebook.py` 用于维护 Notebook 源码，重新运行会清除已有 Notebook 输出，日常运行无需调用它。
