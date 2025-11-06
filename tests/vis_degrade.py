# tests/vis_degrade.py
import os, torch
import pathlib
from torchvision.utils import make_grid, save_image
from data.sr_dataset import SRHRDataset

# 上移到根目录
os.chdir(pathlib.Path(__file__).resolve().parents[1])
os.makedirs("debug_vis", exist_ok=True)
ds = SRHRDataset(["data/DIV2K/HR/"], crop_size=256)

imgs = []
for i in range(4):
    sample = ds[i]
    lq, hr = sample["lq"], sample["hr"]
    imgs += [lq, hr]
grid = make_grid(torch.stack(imgs, dim=0), nrow=2)  # 两列：左LQ右HR
save_image(grid, "tests/debug_vis/degrade_grid.png")
print("Saved to tests/debug_vis/degrade_grid.png")
