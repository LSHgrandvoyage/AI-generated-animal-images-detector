import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from metrics import compute_metrics
import timm

def get_model(model_name, num_classes=2):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name == 'vit_base':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        return model


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir, model_name):
    save_path = os.path.join(save_dir, f"{model_name}.pth")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels, all_preds, all_probs = [], [], []

        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_pbar = tqdm(train_loader, desc=f"Training", unit="batch")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

            train_pbar.set_postfix({'loss': loss.item()})


        epoch_loss = running_loss / len(train_dataset)
        epoch_metrics = compute_metrics(all_labels, all_preds, all_probs)
        print(f"Train Loss: {epoch_loss:.4f}, Train F1: {epoch_metrics['f1-score']:.4f}")

        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation F1: {val_metrics['f1-score']:.4f}, Validation AUC: {val_metrics['roc-auc']:.4f}")

    torch.save(model.state_dict(), save_path) #save 파일명 추후 변경
    return save_path

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        val_pbar = tqdm(loader, desc=f"Validation", unit="batch")
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

        val_loss = running_loss / len(loader.dataset)
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        return val_loss, metrics

def load_model(model, path, device):
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    print(f"Model is loaded successfully! Model : {path}")
    return model

def predict(model, image_path, device, class_names, transform):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    print(f"Prediction: {class_names[pred.item()]} {conf.item() * 100:.2f}%")
    return class_names[pred.item()], conf.item()

load_dotenv()
data_dir = os.getenv("DATA_PATH")
save_dir = os.getenv("SAVE_PATH")
model_name = os.getenv("MODEL_NAME", 'resnet50')

batch_size = 32
num_epochs = 10
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transforms)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transforms)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes

model = get_model(model_name, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

if __name__ == "__main__":
    save_path = os.path.join(save_dir, f"{model_name}.pth")

    if not os.path.exists(save_path):
        save_path = train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir, model_name)

    model = load_model(model, save_path, device)

    test_img = os.path.join(data_dir, 'singletest', class_names[0], 'testdog.jpg')
    predict(model, test_img, device, class_names, transforms)

