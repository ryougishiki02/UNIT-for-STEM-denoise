
#python inspect_checkpoints.py
#访问logs  tensorboard --logdir="D:\huangziyang\wuyucong\UNIT/logs/unit_STM_folder"

import torch

def load_and_inspect_checkpoint(filepath):
    checkpoint = torch.load(filepath,weights_only=True)
    for key, value in checkpoint.items():
        print(f"{key}: {type(value)}")  # 显示每个存储项的键和类型
        if isinstance(value, dict):      # 如果是字典，则显示键
            print("Contents:", list(value.keys()))
            # 查看 state 和 param_groups 的详细信息
            #if 'state' in value:
            #    print("State contents:", value['state'])
            if 'param_groups' in value:
                for group in value['param_groups']:
                    print("Param group:", group)


# 用实际路径替换这些路径（如果需要）
checkpoint_files = [
    'outputs/unit_STM_folder\checkpoints/dis_00000200.pt',
    'outputs/unit_STM_folder\checkpoints/gen_00000200.pt',
    'outputs/unit_STM_folder\checkpoints/optimizer.pt'
]

for file in checkpoint_files:
    print(f"\nInspecting {file}:")
    load_and_inspect_checkpoint(file)