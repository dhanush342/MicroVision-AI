import torch
import torch.nn as nn
import timm
from torchvision import models

def get_efficientnet(num_classes):
    model = timm.create_model("efficientnet_b3", pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def get_resnet(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model