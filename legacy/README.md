# legacy：notebook 时代产物（仅存档）

这里的文件**不参与**新工具链，新代码不 import 它们。留着是为了可追溯，不是可运行。

| 路径 | 原用途 | 为什么不沿用 |
|---|---|---|
| `notebook.ipynb` | 课程 Notebook：讲解 + 调用 `start_lab()` 一步步跑 | 一个 174KB 文件里塞讲解、代码、base64 图和输出；状态藏在内核里，无法批跑和对照 |
| `doodlelab_notebook/` | 旧包：`studio.py` 用字符串拼 HTML/CSS，`panels.py` 用 ipywidgets 做按钮 | 输出、排版、交互与训练逻辑混在一起；`display()` 只能在 notebook 内有意义 |
| `docs/` | 面向 notebook 的学生说明（含 anaconda/requirements 安装步骤） | 安装方式、目录名（`资源/`）都与现状不符 |
| `requirements-common.txt` | 旧的 pip 依赖清单（torch 2.7.0 CPU、numpy 1.26.4…） | 由 `pyproject.toml` + `uv.lock` 取代 |
| `runs_from_notebook/` | 两次旧实验（`第一次训练`、`学习率对照`） | 旧记录格式（依赖 `实验记录.txt`/`model.pt` 内嵌 record），新 CLI 会跳过并在 `runs` 中提示 |

需要旧实验的结论：直接读 `runs_from_notebook/*/实验记录.txt`。
需要在新工具链里重现它们：用 README 里的命令按同样条件重跑一遍即可（默认 MLP、12 轮、lr 1e-3）。
旧权重不再能直接加载——`model.pt` 的结构约定已随工具链调整。
