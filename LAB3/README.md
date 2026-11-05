# 实验三：CIFAR-10 图像分类

朱昕宇，学号 10245501426。

本项目实现基础 CNN、随机裁剪与翻转的数据增强对照，以及带残差连接和通道注意力的改进 CNN。训练时从官方训练集划分验证集，使用验证集选择检查点，最后评估官方测试集。代码不使用预训练权重。

## 安装

需要 Python 3.10–3.12，推荐 Python 3.12；先用 `python --version` 确认版本。以下命令在本文件所在目录执行。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

有兼容 NVIDIA 显卡的环境，先安装 CUDA 12.4 版 PyTorch，再安装其余依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

仅使用 CPU：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

所用 PyTorch 与 torchvision 版本组合参照 [PyTorch 官方版本安装说明](https://pytorch.org/get-started/previous-versions/)。CIFAR-10 来自[官方数据集页面](https://www.cs.toronto.edu/~kriz/cifar.html)，首次运行自动下载。原始数据有 50,000 张训练图和 10,000 张测试图，每张为 32×32 彩色图像，共 10 类。

如果官方数据下载较慢，可先用附带工具并行下载同一个官方压缩包；该工具支持复用完整分片，并在合并后核对官方 MD5。训练时会自动解压已下载的压缩包：

```powershell
.\.venv\Scripts\python.exe tools/download_data.py --data-dir ../data --workers 4
```

## 复现实验

运行三组完整训练，然后分别在测试集评估：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Python .\.venv\Scripts\python.exe -Epochs 30 -Device cuda
```

脚本依次训练三组，全部训练结束后才执行测试评估。正式实验统一采用 30 轮、batch 128、AdamW（学习率 0.001、权重衰减 0.0005）、余弦学习率、seed 42，CUDA 自动混合精度及 channels-last 内存布局；标签平滑和早停关闭：

| 输出目录 | 模型 | 训练数据增强 | 用途 |
| --- | --- | --- | --- |
| `results/baseline` | 基础 CNN | 无 | 基准 |
| `results/ablation` | 基础 CNN | 随机裁剪、随机水平翻转 | 观察增强影响 |
| `results/improved` | 残差与通道注意力 CNN | 随机裁剪、随机水平翻转 | 改进模型 |

默认使用 45,000 张图像训练、5,000 张图像验证，官方 10,000 张测试图不参与训练、调参或早停。所有实验使用相同划分和随机种子。训练与验证数据使用独立的图像变换，验证、测试不启用随机增强。像素转为 `[0, 1]` 后，按通道均值 `(0.4914, 0.4822, 0.4465)`、标准差 `(0.2470, 0.2435, 0.2616)` 标准化。

仅运行一组：

```powershell
.\.venv\Scripts\python.exe -m src.train --model improved --augment standard --epochs 30 --data-dir ../data --output-dir results/improved --device cuda --channels-last
.\.venv\Scripts\python.exe -m src.evaluate --checkpoint results/improved/best.pt --data-dir ../data --output-dir results/improved --device cuda
```

CPU 可将 `--device cuda` 改为 `--device cpu`，或将脚本参数改为 `-Device cpu`。若 Windows 多进程数据加载出现问题，保留默认 `--num-workers 0`。完整训练耗时随硬件、供电和散热变化；本项目最初估计安装、下载、训练及报告合计约 1.5–2.5 小时，CPU 通常更久，实际训练时长以各组 `summary.json` 为准。

可用 `--train-limit`、`--val-limit` 做快速流程检查；这会改变数据规模，不能把该结果当作完整训练结果。所有最终报告数字应来自保存的实际评估文件。

## 结果与指标

2026-11-05 在 RTX 4060 Laptop 8GB 上的实际运行结果如下。均为 seed 42、45,000 训练 / 5,000 验证 / 10,000 测试、30 轮；检查点按验证准确率选择：

| 组别 | 最佳验证准确率 | 测试准确率 | 训练耗时 |
| --- | ---: | ---: | ---: |
| A 基础 CNN | 81.88% | 81.34% | 3.23 分钟 |
| B 基础 CNN + 增强 | 81.14% | 81.10% | 5.41 分钟 |
| C 残差 SE-CNN + 增强 | 89.56% | 88.74% | 11.30 分钟 |

三组纯训练合计约 20 分钟，不含首次下载、安装及报告制作。本次增强组未超过基础组，组合架构改进取得较高准确率；单种子及固定训练预算限制了结论范围，不能把组合提升全部归因于 SE。

实际指标保存在每组 `results/<实验名>/test_metrics.json` 中，训练日志为 `history.csv`，训练设置、运行环境和数据划分摘要为 `summary.json`。最终实验表由报告读取这些文件生成，不在 README 中手工复制数字。

每组评估输出：

- `training_curves.png`：训练、验证 loss 与 accuracy 曲线。
- `confusion_matrix.png`：行是真实类别、列是预测类别，格内为样本计数。
- `misclassifications.png`：预测置信度最高的错误样本，附真实类别、预测类别与置信度。
- `gradcam.png`：末级空间卷积特征层的 Grad-CAM 可视化；改进模型使用最后残差块的 `conv2`，避免选择仅有 1×1 空间的注意力门控层。
- `predictions.csv`：逐样本真实标签、预测标签和置信度。
- `test_metrics.json`：总准确率、每类准确率、precision、recall、F1 和完整混淆矩阵。

总准确率为全部预测正确数除以全部测试样本数。报告的“每类准确率”指该真实类别中预测正确的比例，即该类 recall；它与 precision 的分母不同。Grad-CAM 用于观察模型关注区域，不是因果解释，也不能仅据此断言模型理解了目标。

## 代码结构

```text
src/models.py          两个 CNN 模型
src/train.py           数据划分、训练、验证与检查点
src/evaluate.py        测试指标、曲线、误分类与 Grad-CAM
tests/test_experiment.py 核心行为检查
run_experiments.ps1    三组实验入口
tools/build_report.py  从真实结果生成不超过五页的 PDF
tools/download_data.py 官方压缩包并行下载与 MD5 校验
results/               训练日志、评估指标和插图
```

## 报告与提交

报告上限为 **5 页**，格式为 PDF，开头必须填写真实 GitHub 仓库链接。上传本目录内容至自己的 GitHub 仓库后，使用真实地址重新生成报告：

```powershell
.\.venv\Scripts\python.exe tools/build_report.py --repo-url https://github.com/你的用户名/你的仓库名
```

不要将示例网址作为实际地址。没有填写真实仓库链接的版本属于待补全稿，文件名带有 `_待填仓库链接`，不应直接提交。填写链接后，报告默认输出到上一级目录的 `output/pdf/实验三报告_朱昕宇_10245501426_学生视角版.pdf`；生成后再次核对第一页的链接和 PDF 总页数。报告已按学生向老师汇报课堂实验的口吻编写。

上传 GitHub 时保留代码、README、依赖列表、运行脚本和 `results/` 中的 CSV、JSON、PNG。`data/`、`.venv/` 和检查点 `.pt` 文件已在 `.gitignore` 中排除；数据可重新下载，权重可由训练命令重新生成。手动网页上传同样不要选择这些大文件。最后通过课程指定入口上传 PDF。

## 保存结果的独立核验与打包

完整实验结束后执行：

```powershell
python tools/verify_results.py
python tools/package_submission.py
```

前者从每组 10,000 条预测重建混淆矩阵，交叉核验指标、最佳验证轮次、数据划分和检查点哈希，记录为 `results/verification.json`；后者只打包代码、说明和结果证据，不包含数据和模型检查点。

## 核心行为检查

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

这些检查不下载数据、不启动完整训练，覆盖分层切分的无交集与类别数、验证集禁用随机增强、手工可核对的分类指标、两个模型的梯度及 Grad-CAM。训练的随机种子固定，但 GPU 的并行运算可能有小幅非确定性；如需严格限制算子非确定性，训练时可加 `--deterministic`。同一模型和设置的再次训练应使用新的输出目录。

中断恢复时，在**原训练命令**末尾追加 `--resume results/<实验名>/last.pt`。保留原来的模型、`--augment`、`--epochs`、批量、优化器、学习率、标签平滑、`--channels-last` 等全部参数，不能修改原定 epoch 预算。例如，上面的改进模型运行中断后：

```powershell
.\.venv\Scripts\python.exe -m src.train --model improved --augment standard --epochs 30 --data-dir ../data --output-dir results/improved --device cuda --channels-last --resume results/improved/last.pt
```
