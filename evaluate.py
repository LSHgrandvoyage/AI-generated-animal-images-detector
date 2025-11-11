import os
import torch
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from data_loader import get_data_loader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from model_builder import build_model
from utils.metrics import compute_metrics

class GANImageDataset(Dataset):
    def __init__(self, ai_dir, real_dir, transform=None):
        self.transform = transform
        self.files = []
        self.labels = []
        for f in os.listdir(ai_dir):
            if f.endswith('.png'):
                self.files.append(os.path.join(ai_dir, f))
                self.labels.append(0)
        for f in os.listdir(real_dir):
            if f.endswith('.png'):
                self.files.append(os.path.join(real_dir, f))
                self.labels.append(1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

def load_saved_model(model_path, device):
    model_fullname = model_path.split(os.sep)[-1]
    model_name = model_fullname.split('_')[0]
    if model_name != 'resnet50':
        model_name += '_' + str(model_fullname.split('_')[1])

    model = build_model(model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

def evaluate_model(model, model_filename, test_loader, device):
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    metrics = compute_metrics(y_true, y_pred, y_prob)
    info = parse_model_tag(model_filename)

    return {**info, **metrics}

def parse_model_tag(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split("_")

    info = {
        "model": parts[0],
        "optimizer": parts[1],
        "lr": get_value(parts, "lr"),
        "batch": get_value(parts, "batch"),
        "num_epochs": get_value(parts, "epoch"),
    }

    if parts[0] != 'resnet50':
        info['model'] += parts[1]
        info['optimizer'] = parts[2]

    return info

def get_value(parts, key):
    for p in parts:
        if p.startswith(key):
            try:
                return float(p.split("=")[1])
            except ValueError:
                return p.split("=")[1]
    return None

def save_results(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(results)

    csv_path = os.path.join(save_dir, "trash.csv")
    df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
    print(f"Metrics DONE!!!")

if __name__ == '__main__':
    load_dotenv()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = os.getenv('DATA_PATH')
    gan_dir = os.path.join(data_dir, 'test', 'ai', 'style_gan2')
    real_dir = os.path.join(data_dir, 'test', 'real')
    save_dir = os.getenv('SAVE_PATH')

    #_, _, test_loader = get_data_loader(data_dir, batch=32)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    gan_dataset = GANImageDataset(gan_dir, real_dir, transform=transform)
    test_loader = DataLoader(gan_dataset, batch_size=32, shuffle=False)

    models = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
    results = []

    for idx, file in enumerate(tqdm(models, desc="Evaluating models")):
        model_path = os.path.join(save_dir, file)
        model = load_saved_model(model_path, device)
        results.append(evaluate_model(model, file, test_loader, device))
        torch.cuda.empty_cache()
        tqdm.write(f"[{idx + 1}/{len(models)}] Finished evaluating {file}")

    save_results(results, save_dir)
