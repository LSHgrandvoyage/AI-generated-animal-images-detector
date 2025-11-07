import os
import torch
from data_loader import get_data_loader
from model_builder import build_model, build_optimizer
from trainer import train_and_save
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = os.getenv('DATA_PATH')
save_dir = os.getenv('SAVE_PATH', 'results')

model_list = ['resnet50', 'efficientnet_b0', 'vit_base']
optimizer_list = ['adam', 'sgd']
lr_list = [1e-3, 1e-4, 1e-5]
batch_list = [32]
num_epochs_list = [10, 20]

results = []

for model_name in model_list:
    for optimizer_name in optimizer_list:
        for lr in lr_list:
            for batch in batch_list:
                for num_epochs in num_epochs_list:
                    model_tag = f"{model_name}_{optimizer_name}_lr={lr}_batch={batch}_epoch={num_epochs}"
                    model_path = os.path.join(save_dir, f"{model_tag}.pth")
                    if os.path.exists(model_path):
                        tqdm.write(f"Skip {model_tag} (already exists)")
                        continue

                    tqdm.write(f"\n Training {model_name} | {optimizer_name} | lr = {lr} | batch = {batch} | epoch = {num_epochs}")

                    train_loader, val_loader, test_loader = get_data_loader(data_dir, batch)
                    model = build_model(model_name).to(device)
                    optimizer = build_optimizer(optimizer_name, model, lr)


                    temp_result = train_and_save(model, optimizer, train_loader, val_loader, device,
                                                save_dir, model_tag, num_epochs)

                    tqdm.write(f"{temp_result}")
