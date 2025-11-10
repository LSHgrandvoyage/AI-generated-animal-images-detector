import os
import torch
import torch.nn as nn
from tqdm import tqdm

def train_and_save(model, optimizer, train_loader, val_loader, device, save_dir, model_tag, num_epochs):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False) as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_acc = evaluate_val_acc(model, val_loader, device)
        tqdm.write(f"\nEpoch {epoch + 1}/{num_epochs} - Val Acc: {val_acc:.4f}")

    save_path = os.path.join(save_dir, f"{model_tag}.pth")
    torch.save(model.state_dict(), save_path)
    tqdm.write(f"Model saved in {save_path}")

    return {"val_acc_last": val_acc}

def evaluate_val_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
