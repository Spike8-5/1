import os, json, argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import lpips
import warnings
from accelerate import Accelerator
from diffusers import DDPMScheduler
from models.teacher import SD15Teacher
from models.student import build_student
from diffusion.ddpm import DDPMHelper
from data.sr_dataset import SRHRDataset
from models.embedding_generalize import encode_text
from ram.models import ram_plus

def load_or_none(path, model):
    if os.path.exists(path):
        sd = torch.load(path, map_location='cpu')
        model.load_state_dict(sd, strict=False)
        print(f"[Teacher] Loaded weights from {path}")
        return True
    else:
        print(f"[Teacher] No checkpoint found at {path}. Will fallback to EMA self-teacher for debugging.")
        return False

def main(cfg):
    # Dataset / Loader
    train_roots = ["data/DIV2K/HR/"]
    train_set = SRHRDataset(train_roots, crop_size=256)  # 先从256裁剪起
    train_loader = DataLoader(
        train_set,
        batch_size=1,  # 4060 建议 1 或 2，再用梯度累积放大有效batch
        shuffle=True,
        num_workers=4,  # Windows 可先 2~4，若报错再降
        pin_memory=True,
        drop_last=True
    )

    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=8, cpu=False)
    device = accelerator.device

    # Models
    teacher = SD15Teacher(dtype=torch.bfloat16, device=device)

    student = build_student(in_channels=4, base=64).to(device)
    student.train()

    # 定义优化器/损失/噪声调度器
    opt = torch.optim.AdamW(student.parameters(), lr=5e-5, weight_decay=1e-4)
    warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
    warnings.filterwarnings("ignore", message="Arguments other than a weight enum")
    lpips_loss = lpips.LPIPS(net="vgg").to(device).eval()
    mse = nn.MSELoss()
    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    # accelerate 准备
    student, opt, train_loader = accelerator.prepare(student, opt, train_loader)
# ———————————————————————————————————————————————————————————————————————————————————————————
#     os.makedirs(cfg['paths']['student_out'], exist_ok=True)
#     global_step = 0

    lambda_lat = 0.2
    gamma_reg = 1.0

    # 加载RAM
    ram_model = ram_plus(pretrained=r"D:\pycharmproject\multimodal_isr_distill_stage1\pretrained\ram_plus_swin_large_14m.pth", vit='swin_l', image_size=384).eval().to(device)

    for epoch in range(49):
        pbar = tqdm(train_loader, disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(pbar):
            hr = batch["hr"].to(device, dtype=torch.bfloat16)  # [0,1]
            lr = batch["lr"].to(device, dtype=torch.bfloat16)
            cond = batch["condition"].to(device, dtype=torch.bfloat16)

            # 编码到 latent
            with torch.no_grad():
                z_hr = teacher.encode_vae(hr)  # [B,4,h,w]
                z_lr = teacher.encode_vae(lr)

            # 采样时刻/加噪
            b = z_lr.shape[0]
            t = torch.randint(0, noise_scheduler.num_train_timesteps, (b,), device=device).long()
            noise = torch.randn_like(z_lr)
            zt_lr = noise_scheduler.add_noise(z_lr, noise, t)

            # Teacher ε
            with torch.no_grad():
                eps_T = teacher.predict_eps(zt_lr, t, cond)  # [B,4,h,w]

            # Student ε
            eps_S = student(zt_lr, t, cond).sample if hasattr(student(zt_lr, t, cond), "sample") else student(zt_lr, t, cond)


            # 蒸馏正则（最小可行）
            L_reg = mse(eps_S.float(), eps_T.float())

            # 从 ε 还原 z0 以算重建（稳定器）
            step_out = noise_scheduler.step(eps_S.to(z_lr.dtype), t, zt_lr)
            z0_pred = step_out.prev_sample  # [B,4,h,w]

            with torch.no_grad():
                sr = teacher.decode_vae(z0_pred.to(teacher.vae.dtype))  # [B,3,H,W] in [0,1]

            L_rec = lpips_loss(sr.float(), hr.float())

            loss = L_rec + gamma_reg * L_reg + lambda_lat * mse(z0_pred.float(), z_hr.float())

            accelerator.backward(loss)
            opt.step()
            opt.zero_grad(set_to_none=True)

            if accelerator.is_local_main_process and step % 50 == 0:
                pbar.set_description(
                    f"ep{epoch} st{step} loss:{loss.item():.4f} rec:{L_rec.item():.4f} reg:{L_reg.item():.4f}")

        if accelerator.is_local_main_process and (epoch + 1) % 10 == 0:
            accelerator.save(student.state_dict(), os.path.join("checkpoints", "student", f"ckpt_student_ep{epoch + 1}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/distill_base.json")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = json.load(f)
    main(cfg)

# 问题一：生成词嵌入的时间太慢，直接让训练时间翻了六倍
# 问题二：教师模型的condition也要用LR图像生成吗？
#
