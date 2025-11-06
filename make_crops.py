# 固定图像的裁剪模式，返回坐标
import os
import random
from PIL import Image
import json

# 生成12组固定的裁剪方式
def random_crop(img_path, K=12, img_size=256):
    w, h = Image.open(img_path).size
    coords = []
    for _ in range(K):
        x = random.randint(0, w - img_size)
        y = random.randint(0, h - img_size)
        coords.append((x, y, img_size))
    return coords


if __name__ == '__main__':
    random.seed(666)
    root = r"D:\pycharmproject\multimodal_isr_distill_stage1\data\DIV2K\HR"
    img_list = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.png')])
    items = []
    for _, p in enumerate(img_list):
        basename = os.path.splitext(os.path.basename(p))[0]
        for idx, (x, y, img_size) in enumerate(random_crop(p)):
            items.append({"id": basename, "path": p, "x": x, "y": y, "img_size": img_size, "idx":idx})
    json.dump(items, open('crops.json', 'w'))  # 成功创建每张图片的12种随机裁剪坐标的json文件
