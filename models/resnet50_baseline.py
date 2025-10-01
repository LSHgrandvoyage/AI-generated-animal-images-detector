import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir):
    save_path = os.path.join(save_dir, "resnet50.pth")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

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

            _, predicted = torch.max(outputs.data, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

            train_pbar.set_postfix({'loss': loss.item()})


        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_preds / total_preds
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        validate(val_loader)

    torch.save(model.state_dict(), save_path)
    return save_path

def validate(val_loader):
    model.eval()
    val_loss = 0.0
    val_correct_preds = 0
    val_total_preds = 0

    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Validation", unit="batch")
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            val_total_preds += labels.size(0)
            val_correct_preds += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_correct_preds / val_total_preds
        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Acc: {val_epoch_acc:.4f}\n")

def load_model(model, path, device):
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    print(f"Model is loaded successfully!")
    return model

def predict(model, image_path, device, class_names, transform):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    print(f"Prediction: {class_names[pred.item()]} ({conf.item() * 100:.2f}%")
    return class_names[pred.item()], conf.item()

load_dotenv()
data_dir = os.getenv("DATA_PATH")
save_dir = os.getenv("SAVE_PATH")

batch_size = 32
num_epochs = 10
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

class_names = train_dataset.classes

model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2) # Binary classification
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

if __name__ == "__main__":
    save_path = os.path.join(save_dir, "resnet50.pth")

    if not os.path.exists(save_path):
        save_path = train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir)

    model = load_model(model, save_path, device)

    test_img = os.path.join(data_dir, 'val', class_names[0], 'example_01.jpg') # img path must be changed
    predict(model, test_img, device, class_names, val_transforms)
