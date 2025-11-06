# data/sr_dataset.py
import os, glob, random
from torch.utils.data import Dataset
import cv2
from .real_esrgan_degrade import real_esrgan_degrade

class SRHRDataset(Dataset):
    def __init__(self, roots, crop_size=256):
        self.paths = []
        for r in roots:
            self.paths.append(r)
        self.paths.sort()
        self.crop_size = crop_size
        assert len(self.paths) > 0, f"No images found in {roots}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        basename = os.path.split(os.path.basename(p))[0]
        img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(p)
        lr, hr, cond = real_esrgan_degrade(img_bgr, basename, crop_size=self.crop_size)
        return {"lr": lr, "hr": hr, "condition": cond}
