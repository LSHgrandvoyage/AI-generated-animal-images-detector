import os
import torch
import sys
import torch.nn.functional as F
from dotenv import load_dotenv
from torchvision.utils import save_image
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../stylegan2-ada-pytorch-main/stylegan2-ada-pytorch-main"))

import dnnlib
import legacy

load_dotenv()
data_dir = os.path.join(os.getenv("DATA_PATH"), 'test', 'ai', 'style_gan2')
#network_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl" # cat images
#network_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl"  # dog images
network_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl"  # wild animal images
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading GAN...")

with dnnlib.util.open_url(network_url, verbose=False) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

num_images = 2300
for i in range(num_images):
    z = torch.randn(1, G.z_dim, device=device)
    img = G(z, None, truncation_psi=0.7, noise_mode='const')
    img = (img.clamp(-1, 1) + 1) / 2
    img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)

    save_path = os.path.join(data_dir, f"ai_image_{i:04d}.png")
    save_image(img, save_path)
    print(f"Saved : {save_path}")

print("DONE!!!!")