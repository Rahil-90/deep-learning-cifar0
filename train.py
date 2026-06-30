import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader import train_loader
from model import get_model

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load model
model = get_model().to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5

train_losses = []
train_accuracies = []

# Training loop
for epoch in range(epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Loss: {epoch_loss:.4f}")
    print(f"Accuracy: {epoch_accuracy:.2f}%")

# Save model
torch.save(
    model.state_dict(),
    "outputs/cifar10_model.pth"
)

print("\nModel Saved Successfully!")

# Plot training graphs
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()

plt.savefig("outputs/training_curve.png")

plt.show()