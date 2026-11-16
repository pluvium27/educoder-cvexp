"""自定义模型入口：替换构建函数后，训练、评价、保存、加载与图片预测沿用共同流程。

输入 params 是 lab.train(custom_params=...) 的 JSON 可保存字典；num_classes 固定为 8。
返回 CPU torch.nn.Module：输入 float32 [B,1,28,28]、范围 [0,1]，
输出 [B,8] 未做 softmax 的分数（logits）。训练循环负责损失、优化器与评价。
当前示例为小型MLP，性能应通过实际实验评价。
改动后重启内核；每次实验自动保留源码快照，加载旧模型要恢复相应结构。
"""
from torch import nn


def build_custom_model(params, num_classes=8):
    """例如 lab.train('自定义模型', model='custom', custom_params={'width':48})。"""
    unknown=set(params)-{'width'}
    if unknown: raise ValueError(f'当前示例仅接受width参数，收到{unknown}；自定义时同步修改此检查。')
    width=params.get('width',48)
    if not isinstance(width,int) or width<1: raise ValueError('width需正整数')
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,width), nn.ReLU(),
        nn.Linear(width,24), nn.ReLU(),
        nn.Linear(24,num_classes),
    )
