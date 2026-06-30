import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models

# CIFAR10 Classes
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load Model
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    10
)

model.load_state_dict(
    torch.load(
        "outputs/saved model file.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Image
image = Image.open(
    "test_image.jpeg"
).convert("RGB")

image = transform(image)

image = image.unsqueeze(0).to(device)

# Prediction
with torch.no_grad():

    outputs = model(image)

    _, predicted = torch.max(outputs, 1)

predicted_class = classes[predicted.item()]

print(f"Predicted Class: {predicted_class}")