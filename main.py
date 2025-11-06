import os
import torch
import pandas as pd
from data_loader import get_data_loader
from model_builder import build_model, build_optimizer
from trainer import train_and_evaluate
from dotenv import load_dotenv

load_dotenv()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = os.getenv('DATA_PATH')
save_dir = os.getenv('SAVE_PATH', 'results')

model_list = ['resnet50', 'efficientnet_b0', 'vit_base']
optimizer_list = ['adam', 'sgd']
lr_list = [1e-3, 1e-4, 1e-5]
batch_list = [32, 64]
num_epochs_list = [10, 15, 20]

results = []

for model_name in model_list:
    for optimizer_name in optimizer_list:
        for lr in lr_list:
            for batch in batch_list:
                for num_epochs in num_epochs_list:
                    print(f"\n Training {model_name} | {optimizer_name} | lr = {lr} | batch = {batch}")

                    train_loader, val_loader, test_loader = get_data_loader(data_dir, batch)
                    model = build_model(model_name).to(device)
                    optimizer = build_optimizer(optimizer_name, model, lr)

                    model_tag = f"{model_name}_{optimizer_name}_lr={lr}_batch={batch}_epoch={num_epochs}"
                    test_metrics = train_and_evaluate(model, optimizer, train_loader, val_loader, test_loader,
                                                      device, save_dir, model_tag, num_epochs)
                    results.append({
                        "model": model_name,
                        "optimizer": optimizer_name,
                        "lr": lr,
                        "batch": batch,
                        'num_epochs': num_epochs,
                        **test_metrics
                    })

df = pd.DataFrame(results)
csv_path = os.path.join(os.getenv('SAVE_PATH'), 'results.csv')
df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
print(f"\n DONE~~~~~~~~~~~~~~")