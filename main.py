import torch
import torch.nn as nn

from torchvision import models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import wandb
import os, csv, sys
from PIL import Image


os.environ["TORCH_HOME"] = "/ocean/projects/cis240109p/abollado/.cache"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to fit model input
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# Load dataset
train_dataset = ImageFolder(root="dataset/train", transform=transform)
val_dataset = ImageFolder(root="dataset/val", transform=transform)


import torch
import torch.nn as nn
from torchvision import models


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, non_trainable


def load_model(name):

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    elif name == "resnet152":
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    elif name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif name == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    # elif name == "densenet121":
    #     model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    # elif name == "mobilenet_v2":
    #     model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # elif name == "inception_v3":
    #     model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Model {name} not recognized")

    # # Freeze all layers except the final fully connected layer
    # for param in model.parameters():
    #     param.requires_grad = False

    # Modify the last fully connected layer for binary classification
    if hasattr(model, 'fc'):  # ResNet, VGG, DenseNet, etc.
        model.fc = nn.Linear(model.fc.in_features, 1)
    # elif hasattr(model, 'classifier'):  # VGG, MobileNet, Inception
    #     model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
    # elif hasattr(model, 'top'):  # Inception V3
    #     model.top = nn.Linear(model.top.in_features, 1)
    else:
        raise ValueError(f"Model {name} does not have a recognizable final layer.")

    # # Unfreeze the final layer
    # for param in model.fc.parameters() if hasattr(model, 'fc') else model.classifier[6].parameters():
    #     param.requires_grad = True

    trainable, non_trainable = count_parameters(model)
    print(f"Model {name} has {trainable} trainable parameters and {non_trainable} non-trainable parameters")
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



def test(model, test_folder="dataset/test_images", output_csv="submission.csv"):
    # Set model to evaluation mode
    model.eval()

    # Create a list to store the results
    results = []

    # Iterate through test images in the specified folder
    for img_name in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')  # Convert to RGB if needed
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
        
        # Predict label
        with torch.no_grad():
            output = model(image)
            predicted_label = (torch.sigmoid(output) > 0.5).float().item()  # Apply sigmoid and threshold

        # Add image name and predicted label to results
        results.append([img_name, int(predicted_label)])

    # Write the results to a CSV file
    with open(output_csv, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["img_name", "label"])  # Write header
        writer.writerows(results)  # Write predictions

    print(f"Test predictions saved to {output_csv}")



if __name__ == "__main__":

    # get model
    name = sys.argv[1]
    model = load_model(name)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # get data 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # train model
    wandb.init(
        project="cnn-against-malaria",
        name=name,    
    )
    train(model, train_loader, val_loader, criterion, optimizer, epochs=10)
    wandb.finish()

    test(model, output_csv=f"submission_{name}.csv")
