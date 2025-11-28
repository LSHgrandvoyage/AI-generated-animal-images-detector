import os
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return img, tensor

# For CNN(EfficientNet, ResNet)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        self.activations = out

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        output = self.model(x)
        class_idx = torch.argmax(output, 1)
        self.model.zero_grad()
        output[:, class_idx].backward()

        grads = self.gradients[0].detach().cpu().numpy()
        acts = self.activations[0].detach().cpu().numpy()
        weights = grads.mean(axis=(1,2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for w, a in zip(weights, acts):
            cam += w*a

        cam = np.maximum(cam, 0)
        cam /= cam.max() + 1e-8
        return cam

class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = []

        for blk in self.model.blocks:
            blk.attn.register_forward_hook(self.hook)

    def hook(self, module, inp, out):
        # out: [B, num_heads, N, N] or [B, N, N]
        if out.dim() == 4:
            attn = out.mean(1)  # head 평균
        else:
            attn = out
        self.activations.append(attn.detach())

    def __call__(self, x):
        self.activations = []
        _ = self.model(x)

        attn = self.activations[-1][0]  # [N, N]
        cls_attn = attn[0, 1:]          # CLS -> patches

        L = int(np.sqrt(cls_attn.shape[0]))
        if L*L == cls_attn.shape[0]:
            heatmap = cls_attn.reshape(L,L).cpu().numpy()
        else:
            heatmap = cls_attn.cpu().numpy()

        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap

def detect_model_type(filename):
    name = filename.lower()
    if name.startswith("resnet50"):
        return "resnet50"
    elif name.startswith("efficientnet_b0"):
        return "efficientnetb0"
    elif name.startswith("vit_base"):
        return "vitbase"
    else:
        raise ValueError(f"Unknown model type: {filename}")

def load_model(model_path, model_type):
    if model_type == "resnet50":
        model = timm.create_model('resnet50', pretrained=False, num_classes=2)
        target_layer = model.layer4[-1]

    elif model_type == "efficientnetb0":
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
        target_layer = model.blocks[-1]

    elif model_type == "vitbase":
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        target_layer = None

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, target_layer

def visualize_and_save(model, model_type, target_layer, img, tensor, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)

    if model_type in ["resnet50", "efficientnetb0"]:
        cam = GradCAM(model, target_layer)(tensor)
        cam_img = Image.fromarray(np.uint8(cam*255)).resize(img.size)
        cam_arr = np.array(cam_img)

        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.imshow(cam_arr, alpha=0.5, cmap='jet')
        plt.axis('off')

    else:  # ViT
        heatmap = ViTGradCAM(model)(tensor)
        if heatmap.ndim == 1:
            heatmap_img = Image.fromarray(np.uint8(heatmap*255)).resize(img.size)
            heatmap = np.array(heatmap_img)
        else:
            heatmap = (heatmap*255).astype(np.uint8)
            heatmap_img = Image.fromarray(heatmap).resize(img.size)
            heatmap = np.array(heatmap_img)

        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.imshow(heatmap, alpha=0.5, cmap='jet')
        plt.axis('off')

    save_path = os.path.join(save_dir, filename + ".png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    load_dotenv()
    MODEL_DIR = os.getenv("SAVE_PATH")
    IMG_PATH = os.path.join(os.getenv("SINGLE_IMAGE_PATH"), 'real.png') # single image
    SAVE_DIR = os.path.join(os.getcwd(), "real_heatmap") # or 'ai_heatmap'

    img, tensor = load_image(IMG_PATH)

    for file in os.listdir(MODEL_DIR):
        if file.endswith(".pth"):
            model_path = os.path.join(MODEL_DIR, file)
            model_type = detect_model_type(file)
            model, target_layer = load_model(model_path, model_type)
            visualize_and_save(model, model_type, target_layer, img, tensor, SAVE_DIR, file)
