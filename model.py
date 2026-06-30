import torch.nn as nn
from torchvision import models

def get_model():

    # Load pretrained ResNet18
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Replace output layer
    model.fc = nn.Linear(
        model.fc.in_features,
        10
    )

    return model