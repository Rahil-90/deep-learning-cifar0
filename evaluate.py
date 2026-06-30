import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from data_loader import test_loader
from model import get_model

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load model
model = get_model().to(device)

model.load_state_dict(
    torch.load(
        "outputs/cifar10_model.pth",
        map_location=device
    )
)

model.eval()

all_predictions = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

# Accuracy
accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")

# Classification Report
print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions
    )
)

# Confusion Matrix
cm = confusion_matrix(
    all_labels,
    all_predictions
)

# Plot confusion matrix
plt.figure(figsize=(8, 6))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.colorbar()

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.show() 