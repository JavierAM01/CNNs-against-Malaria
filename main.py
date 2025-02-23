import torch
import torch.nn as nn

from torchvision import models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import wandb


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to fit model input
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# Load dataset
train_dataset = ImageFolder(root="dataset/train_images", transform=transform)
val_dataset = ImageFolder(root="dataset/val", transform=transform)


def load_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Modify last layer for binary classification
    model = model.to(device)    
    return model

def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)  # Convert labels to float
            labels = labels.view(-1, 1)  # Reshape for BCEWithLogitsLoss

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # Convert logits to 0/1
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = evaluate(model, val_loader)

        wandb.log({
            "train_loss": total_loss / len(train_loader),
            "train_acc": train_acc,
            "val_acc": val_acc
        }, step=epoch)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}, Val Acc: {val_acc:.2f}")

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().to(device)
            labels = labels.view(-1, 1)

            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total




if __name__ == "__main__":

    # get model
    model = load_model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # get data 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # train model
    wandb.init(
        project="cnn-against-malaria",
        name="resnet18",    
    )
    train(model, train_loader, val_loader, criterion, optimizer, epochs=10)
    wandb.finish()