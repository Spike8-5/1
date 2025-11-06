# data/real_esrgan_degrade.py
import random, io, math
import numpy as np
import cv2
import pickle
from PIL import Image
from torchvision.transforms.functional import to_tensor

# ---------- 1) 随机核生成（各向同性/各向异性高斯） ----------
def random_gaussian_kernel(ksize_range=(7, 21), sigma_range=(0.2, 3.0), anisotropic_prob=0.3):
    k = random.randrange(ksize_range[0] | 1, ksize_range[1] | 1, 2)  # 保证奇数核
    if random.random() < anisotropic_prob:
        # 各向异性：构造协方差矩阵
        sigma_x = random.uniform(*sigma_range)
        sigma_y = random.uniform(*sigma_range)
        theta = random.uniform(0, math.pi)
        # 生成网格
        ax = np.arange(-k//2 + 1., k//2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        c, s = math.cos(theta), math.sin(theta)
        xr = c*xx + s*yy
        yr = -s*xx + c*yy
        kernel = np.exp(-(xr**2/(2*sigma_x**2) + yr**2/(2*sigma_y**2)))
    else:
        # 各向同性
        sigma = random.uniform(*sigma_range)
        ax = np.arange(-k//2 + 1., k//2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)

def filter2D(img, kernel):
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT_101)

# ---------- 2) 随机下采样比例 ----------
def random_resize(img, sf_range=(1.0, 4.0)):
    # sf >=1 表示先下采样（更小），再上采样回目标尺寸
    h, w = img.shape[:2]
    sf = random.uniform(*sf_range)
    new_h, new_w = int(h / sf), int(w / sf)
    new_h = max(16, new_h); new_w = max(16, new_w)
    img_small = cv2.resize(img, (new_w, new_h), interpolation=random.choice(
        [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC]
    ))
    img_up = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return img_up, sf

# ---------- 3) 噪声与压缩 ----------
def add_gaussian_noise(img, sigma_range=(0, 8)):
    sigma = random.uniform(*sigma_range)
    if sigma <= 0:
        return img
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def add_poisson_noise(img, scale_range=(0.0, 0.01)):
    scale = random.uniform(*scale_range)
    if scale <= 0:
        return img
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    out = np.random.poisson(img.astype(np.float32) * scale * vals) / (scale * vals)
    return np.clip(out, 0, 255).astype(np.uint8)

def jpeg_compress(img, q_range=(40, 95)):
    q = random.randint(*q_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    ok, enc = cv2.imencode('.jpg', img, encode_param)
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec

# ---------- 4) 主退化流程 ----------
def real_esrgan_degrade(hr_img_bgr, crops, crop_size=256):
    # 随机中心裁剪（保障尺寸）
    # h, w = hr_img_bgr.shape[:2]
    rand_num = random.randint(0, 11)
    # with open(f"cache/{basename}/crop_{rand_num}.pkl", "rb") as f:
    #     data = pickle.load(f)
    # crop = data["xywh"]
    x1 = crops[0]
    y1 = crops[1]
    x2 = crops[2]
    y2 = crops[3]
    hr = hr_img_bgr[y1:y2, x1:x2].copy()

    # 1) 模糊
    ker = random_gaussian_kernel()
    lq = filter2D(hr, ker)

    # 2) 随机下采样/上采样
    lq, sf = random_resize(lq, sf_range=(1.5, 4.0))

    # 3) 噪声（高斯/泊松）
    if random.random() < 0.7:
        lq = add_gaussian_noise(lq, sigma_range=(0, 10))
    if random.random() < 0.3:
        lq = add_poisson_noise(lq, scale_range=(0.0, 0.01))

    # 4) JPEG 压缩伪影
    if random.random() < 0.8:
        lq = jpeg_compress(lq, q_range=(45, 90))

    # 5) 色彩抖动（轻量，可选）
    # 这里先略过，保持简单稳定

    # 转 tensor（0-1）
    hr_t = to_tensor(Image.fromarray(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)))
    lq_t = to_tensor(Image.fromarray(cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)))

    return lq_t, hr_t

def inf_degrade(hr):
    # 1) 模糊
    ker = random_gaussian_kernel()
    lr = filter2D(hr, ker)

    # 2) 随机下采样/上采样
    sf = random.uniform(1.5, 4.0)
    h, w = hr.shape[:2]
    lr_small = cv2.resize(lr, (int(w / sf), int(h / sf)), interpolation=cv2.INTER_AREA)
    lr = cv2.resize(lr_small, (w, h), interpolation=cv2.INTER_CUBIC)

    # 3) 噪声（高斯/泊松）
    if random.random() < 0.7:
        lr = add_gaussian_noise(lr, sigma_range=(0, 10))
    if random.random() < 0.3:
        lr = add_poisson_noise(lr, scale_range=(0.0, 0.01))

    # 4) JPEG 压缩伪影
    if random.random() < 0.8:
        lq = jpeg_compress(lr, q_range=(45, 90))

    # 5) 色彩抖动（轻量，可选）
    # 这里先略过，保持简单稳定

    # 转 tensor（0-1）
    hr_t = to_tensor(Image.fromarray(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)))
    lq_t = to_tensor(Image.fromarray(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)))

    return lq_t, hr_t