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
data_dir = os.path.join(os.getenv("OLD_DATA_PATH"), 'temp', 'style_gan2')

network_urls = {
    "afhq_cat": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl", # cat images
    "afhq_dog": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl",  # dog images
    "afhq_wild": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl"  # wild animal images
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_images = 10

for model_name, network_url in network_urls.items():
    print(f"Loading GAN... : {model_name}")
    with dnnlib.util.open_url(network_url, verbose=False) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    save_dir = os.path.join(data_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_images):
        z = torch.randn(1, G.z_dim, device=device)
        img = G(z, None, truncation_psi=0.7, noise_mode='const')
        img = (img.clamp(-1, 1) + 1) / 2
        img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)

        save_path = os.path.join(save_dir, f"{model_name}_{i:04d}.png")
        save_image(img, save_path)

    print(f"{model_name} finished!")

print("DONE!!!!")