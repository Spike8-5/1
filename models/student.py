# models/student.py
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

def build_student(in_channels=4, base=64, cross_attention_dim=768, attention_head_dim=8):  # base可调：64/48/32
    model = UNet2DConditionModel(
        sample_size=None,          # 任意分辨率
        in_channels=in_channels,   # latent 通道=4
        out_channels=in_channels,  # 预测 eps 形状一致
        layers_per_block=2,
        block_out_channels=(base, base*2, base*4),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),   # 末层加注意力
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=attention_head_dim,
        norm_num_groups=16,
    )
    # 可选：xFormers
    try:
        model.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return model
