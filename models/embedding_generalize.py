import torch
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import CLIPTokenizer, CLIPTextModel


# 生成图像标签
def inf_ram(img, model, crops, img_size=384):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ram_plus(pretrained=pretrained, vit='swin_l', image_size=img_size)
    model.eval().to(device)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
    ])
    img = transform(img).to(device).unsqueeze(0)
    res = inference(img.float(), model)
    return res


# 生成标签对应的embedding
def encode_text(img, model, crops):
    tags = inf_ram(img, model, crops)
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to("cuda")
    tags = ",".join(str(tags).split(" | "))
    tokens = tokenizer(tags, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to("cuda")
    outputs = text_encoder(**tokens)
    emb = outputs.last_hidden_state
    return emb

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='Tag2Text inferece for tagging and captioning')
#     parser.add_argument('--image',
#                         metavar='DIR',
#                         help='path to image',
#                         default='data/DIV2K_VAL/HR/0801.png')
#     parser.add_argument('--pretrained',
#                         metavar='DIR',
#                         help='path to pretrained model',
#                         default='pretrained/ram_plus_swin_large_14m.pth')
#     parser.add_argument('--img_size',
#                         default=384,
#                         type=int,
#                         metavar='N',
#                         help='input image size (default: 448)')
#     args = parser.parse_args()
#     res = inf_ram(args.image, args.pretrained, args.img_size)
#     print("Image Tags: ", res[0])
#     print("图像标签: ", res[1])
