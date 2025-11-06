# 生成固定裁剪的图片的lr版本的词嵌入

import torch
import json
import os
import pickle
import cv2
from ram.models import ram_plus
from models.embedding_generalize import encode_text
from data.real_esrgan_degrade import real_esrgan_degrade

device = "cuda"
items = json.load(open("crops.json"))
ram_model = ram_plus(
    pretrained=r"D:\pycharmproject\multimodal_isr_distill_stage1\pretrained\ram_plus_swin_large_14m.pth", vit='swin_l', image_size=384).eval().to(device)
for item in items:
    p = item["path"]
    hr_img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)  # hr图像的ndarray
    crops = [item["x"], item["y"], item["x"] + item["img_size"], item["y"] + item["img_size"]]
    lr_img, _ = real_esrgan_degrade(hr_img_bgr, crops)
    emb = encode_text(lr_img, ram_model, crops).unsqueeze(0)  # [1, 77, 768]
    out_dir = f"cache/{item['id']}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/crop_{item['idx']}.pkl", "wb") as f:
        pickle.dump({
            "xywh": (item["x"], item["y"], item["x"] + item["img_size"], item["y"] + item["img_size"]),
            "path": p,
            "text_emb": emb.squeeze(0),  # [77,768]
        }, f)
    print(f"{item['id']},{item['idx']} is done")
