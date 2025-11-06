# models/teacher.py
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


class SD15Teacher:
    def __init__(self, name="runwayml/stable-diffusion-v1-5", dtype=torch.bfloat16, device="cuda"):
        self.device = device
        self.dtype = dtype

        self.vae = AutoencoderKL.from_pretrained(name, subfolder="vae", torch_dtype=dtype).to(device)
        self.unet = UNet2DConditionModel.from_pretrained(name, subfolder="unet", torch_dtype=dtype).to(device)

        # 需要给 UNet 一个“空提示”的文本条件（encoder_hidden_states）
        self.tokenizer = CLIPTokenizer.from_pretrained(name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(name, subfolder="text_encoder", torch_dtype=dtype).to(device)

        # 省显存
        self.vae.enable_slicing()
        self.vae.enable_tiling()
        try:
            self.unet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        # 冻结
        for m in [self.vae, self.unet, self.text_encoder]:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    # @torch.no_grad()
    # def txt_cond(self, batch_size: int):
    #     tokens = self.tokenizer([""] * batch_size, padding="max_length", max_length=self.tokenizer.model_max_length,
    #                             return_tensors="pt")
    #     emb = self.text_encoder(tokens.input_ids.to(self.device))[0]
    #     return emb  # [B, 77, 768]

    @torch.no_grad()
    def encode_vae(self, img):  # img: [B,3,H,W] in [0,1]
        latents = self.vae.encode(img).latent_dist.sample() * 0.18215
        return latents

    @torch.no_grad()
    def decode_vae(self, latents):  # latents: [B,4,h,w]
        img = self.vae.decode(latents / 0.18215).sample.clamp(0, 1)
        return img

    @torch.no_grad()
    def predict_eps(self, zt, t, cond):
        # 兼容 diffusers 的返回
        out = self.unet(zt, t, encoder_hidden_states=cond).sample
        return out
