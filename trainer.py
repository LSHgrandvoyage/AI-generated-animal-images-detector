import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.metrics import compute_metrics

def train_and_evaluate(model, optimizer, train_loader, val_loader, test_loader, device, save_dir, model_tag, num_epochs):
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Val Acc: {val_metrics['accuracy']:.4f}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), os.path.join(save_dir, f'{model_tag}.pth'))

    test_metrics = evaluate(model, test_loader, device)
    return test_metrics

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    return compute_metrics(y_true, y_pred, y_prob)
