import torch.nn as nn
import torch.optim as optim
from torchvision import models
import timm

def build_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        return model

    if model_name == 'vit_base':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        return model

    raise ValueError("MODEL ERROR!!!!!!!!!!!!!!!!!")

def build_optimizer(optimizer_name, model, lr):
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)

    if optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    raise ValueError("OPTIMIZER ERROR!!!!!!!!!!!!!!!")
