import os, argparse, glob
import torch
from PIL import Image
import torchvision.transforms as T
from diffusion.ddpm import DDPMHelper
from utils.common import save_image_grid
from models.student import build_student
from data.real_esrgan_degrade import inf_degrade

def load_img(path):
    im = Image.open(path).convert('RGB')
    return T.ToTensor()(im)

def main(args):
    device = 'cuda'
    model = build_student().to(device)
    sd = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(sd, strict=False)
    model.eval()
    ddpm = DDPMHelper(timesteps=1000, device=device)

    test_roots = ["data/DIV2K_VAL/HR/"]
    os.makedirs(args.out, exist_ok=True)

    for filename in os.listdir(test_roots):
        img_path = os.path.join(test_roots, filename)
        model_dtype = next(model.parameters()).dtype
        hr = load_img(img_path).to(device, dtype=model_dtype)
        lr = inf_degrade(hr).to(device, dtype=model_dtype)

        # 计算输出尺寸
        H_lr, W_lr = lr.shape[1], lr.shape[2]
        H_hr, W_hr = H_lr * args.scale, W_lr * args.scale

        # 上采样LR作为条件，并归一化到[-1,1]
        lr_up = T.Resize((H_hr, W_hr), interpolation=T.InterpolationMode.BICUBIC)(lr)
        lr_up = (lr_up * 2 - 1).unsqueeze(0).to(device)
        shape = (1, 3, H_hr, W_hr)

        with torch.no_grad():
            rec = ddpm.sample(model, shape, cond_img=lr_up, steps=args.steps)  # [1,3,H,W] in [-1,1]

        # 每张单独保存（不再stack）
        base = os.path.splitext(os.path.basename(lr))[0]
        save_path = os.path.join(args.out, f"{base}.png")
        save_image_grid(rec, save_path, nrow=1)
        print(f"Saved: {save_path}")

    print(f"All done. Outputs in: {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    # parser.add_argument("--lr_dir", type=str, required=True)
    parser.add_argument("--mk_dir", required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--max_n", type=int, default=8)
    args = parser.parse_args()
    main(args)
