# AI你画我猜 · 涂鸦识别实验工具链

28×28 八类涂鸦（猫 / 兔子 / 鸟 / 鱼 / 苹果 / 香蕉 / 蛋糕 / 甜甜圈）的**命令行**训练与对照实验工具。
纯 CPU、无 Jupyter、无 notebook 输出、无隐藏全局状态：一条命令训练一次实验，产物落在 `runs/` 里，
`compare` 直接给出带基线的对照表。

## 安装

需要 [uv](https://docs.astral.sh/uv/)。仓库里 `pyproject.toml` 就是完整环境定义；旧 `uv.lock` 已作废
（它锁的是 CUDA 版 torch），先重新生成：

```bash
uv lock          # 解析依赖（torch 只走 CPU 专用源，不再拉 nvidia-* 那 3GB）
uv sync          # 建/更新 .venv
uv run pytest    # 冒烟测试：数据能读、三种模型能建、一次训练可复现
```

- Python 版本由 `.python-version`（3.13）决定；`requires-python = ">=3.11"`。
- `torch` 只从 `pytorch-cpu` 索引取，**不要**再装 CUDA 版：本项目全程 `device='cpu'`。
- 不再需要 `requirements-common.txt`、Anaconda、`pip install` 任何东西（旧文件已归档到 `legacy/`）。

## 目录

```
data/            课程数据 + manifest.json（标签、划分、SHA256）；不联网、不重新下载
src/doodlelab/   工具链源码
runs/            实验产物（git 忽略）：每次实验一个目录
tools/board.html 离线画板，手绘导出 PNG 给 predict 用
legacy/          notebook 时代的旧代码与旧实验，仅存档，不再维护
tests/           冒烟测试
```

## 命令速查

```bash
uv run doodlelab splits                     # 看数据划分与张数
uv run doodlelab models                     # 看可用模型；models resnet 看它的参数

# 1) 基线：小型 MLP
uv run doodlelab train 基线 --model mlp --tag "参照实验"

# 2) 一次只改一个变量：换成 ResNet
uv run doodlelab train 残差 --model resnet \
    --param widths=16,32,64 --param blocks=1,1,1 --tag "残差基线"

# 3) 与基线对照（Δ准确率 + 参数量，写 compare.md/csv/png）
uv run doodlelab compare --baseline 基线 --metric accuracy

# 4) 补做挑战评价（独立原图 / 平移 / 加粗 / 遮挡）
uv run doodlelab eval 残差 --split challenge_original --split challenge_thick

# 5) 用某个实验的模型识别自己的画
uv run doodlelab predict 残差 my_images/doodle.png            # 画板导出（白底黑线）
uv run doodlelab predict 残差 my_images/photo.jpg --mode photo --crop 20,20,400,400
```

其它：`runs`（列表）、`show <实验>`（打印完整记录）、`compare --last 5`、`compare --split challenge_shift`。

`--param` 可重复，值用逗号分隔；参数名写错会被直接拒绝（不是静默忽略）。
每个模型的默认参数见 `uv run doodlelab models <名字>`。

## 一次实验留下什么

`runs/<时间戳>_<实验名>/`：

| 文件 | 内容 |
|---|---|
| `config.json` | 模型与全部训练条件、参数量、耗时、环境（Python/PyTorch 版本）、fingerprint |
| `history.json` | 逐轮 train/val 的 loss 与 accuracy |
| `evaluations.json` | 各划分的准确率、每类表现、混淆矩阵、前 8 条错误 |
| `model.pt` | 权重 + 配置 + 标签 + 预处理约定（`weights_only=True` 读取） |
| `curve.png` | 训练曲线 |
| `summary.md` | 可直接贴给 AI 或抄进报告的纯文本记录 |
| `predictions.md` | 该模型做过的图片识别记录（逐次追加） |

评价口径固定：**保存最后一轮**、不使用早停挑选、验证集可见但不参与训练、没有独立测试集。
种子固定且开启 `torch.use_deterministic_algorithms(True)`；每次训练结束后会自动
「保存 → 重载 → 再评价」，确认分数差为 0，否则报错（防止权重保存链路悄悄坏掉）。

## 做 ResNet 改进 / 新对照实验

模型都在 `src/doodlelab/models.py` 的注册表里，新增结构不用改训练、评价、保存、CLI：

```python
register(ModelSpec(
    name="resnet_wide",                          # 之后 --model resnet_wide 即可
    summary="加宽残差网络",
    params=(
        Param("widths", "int_list", [32, 64, 128], 1, 512, "各 stage 通道"),
        Param("blocks", "int_list", [2, 2, 2], 1, 6, "各 stage 残差块数"),
        Param("stem", "int", 32, 1, 512, "stem 通道"),
        Param("dropout", "float", 0.0, 0.0, 0.9, "块内 dropout"),
    ),
    build=_build_resnet,                          # 复用已有构造函数即可
))
```

对照实验的建议做法：**固定种子与其余条件，只改一个 knob**，然后
`compare --baseline <参照实验>` 看 Δ准确率与参数量倍数，再 `eval` 到挑战划分上验证鲁棒性。

可作为单变量 knobs：`widths` / `blocks` / `stem` / `dropout` / 模型名（mlp↔cnn↔resnet）/
`--lr` / `--weight-decay` / `--label-smoothing` / `--augment-shift` / `--optimizer`。

## 图片输入的预处理（`src/doodlelab/preprocess.py`）

画的图要变成训练数据那种「28×28、黑底亮线」，链路是：
透明合成白底 → 可选裁剪 → 灰度 →（拍照模式自动对比度）→ 中值滤波 → 去掉整齐外框 →
判断墨色方向 → 去掉纸面底色 → 找墨迹范围 → 等比缩放到最长边 26 格 → 居中贴到 28×28。

相对旧版修掉的坑（都能复现）：

1. **总是先判断边框亮度**会把「白纸放在深色桌面上」判反，整张纸被当成墨迹。现在用「墨迹只占少数像素」判断方向，并先裁掉与外框同色的整齐边缘。
2. **中位数找外接框**遇到纸纹/暗角时，会把外接框撑到整张图，图案被缩得很小。现在按「整行/整列墨迹够多」定框，噪点不会撑框。
3. **512 画布缩到 26 格**时，线条只占零点几像素会被插值抹黑。现在按块保底，缩得越狠越保证笔画可见（`--threshold`、`--fit-size` 可调）。

`predict` 会打印实际处理参数并保存「原图 / 模型看到的 28×28」对照图，便于判断问题出在画上还是预处理上。

## 与旧版的关系

`legacy/` 里保留的是改造前的产物：`notebook.ipynb`、`doodlelab_notebook/`（`studio.py` 拼 HTML 字符串 + `panels.py` 的 ipywidgets 面板）、`docs/`（面向 notebook 的学生说明）、
`requirements-common.txt`、`runs_from_notebook/`（旧格式实验）。

它们**不参与**新工具链：新代码不 import 任何 `IPython`/`ipywidgets`，`runs` 遇到旧实验目录会跳过并提示。
需要旧实验的结论时，看 `legacy/runs_from_notebook/*/实验记录.txt`；要复用其中的模型，需按旧 notebook 的方式加载，不再支持。
