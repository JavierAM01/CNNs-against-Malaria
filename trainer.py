import torch
import torch.nn as nn
from torchvision import models

import csv, os
from PIL import Image
import wandb

from data import transform

def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, non_trainable


def get_device(model):
    return next(model.parameters()).device


def load_model(name, device="cpu", pretrained_path=""):

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
    elif name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    elif name == "densenet169":
        model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
    elif name == "mobilenet_v3":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    elif name == "regnet_y_400mf":
        model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1)
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    # elif name == "swin_t":
    #     model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    # elif name == "mnasnet":
    #     model = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1)
    # elif name == "vit_b_16":
    #     model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Model {name} not recognized")

    # Freeze all layers
    for p in model.parameters():
        p.requires_grad = False

    ## RESNET ##
    if "resnet" in name:
        for p in model.fc.parameters():
            p.requires_grad = True
        # n = 256
        # model.fc = nn.Sequential(
        #     nn.Linear(model.fc.in_features, n),  # 2048 -> n
        #     nn.ReLU(),
        #     nn.Linear(n, 1)  # n -> 1
        # )
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),  # 2048 -> n
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        if pretrained_path != "":
            model.fc.load_state_dict(torch.load(pretrained_path))
            print(f"Using model from: {pretrained_path}")

    ## VGG ##
    elif "vgg" in name:
        for p in model.classifier[-1].parameters():
            p.requires_grad = True
        # n = 128
        # model.classifier[-1] = nn.Sequential(
        #     nn.Linear(model.classifier[-1].in_features, n),  # 4096 -> n
        #     nn.ReLU(),
        #     nn.Linear(n, 1)  # n -> 1
        # )
        model.classifier[-1] = nn.Sequential(
            nn.Linear(model.classifier[-1].in_features, 512),  # 4096
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        if pretrained_path != "":
            model.classifier[-1].load_state_dict(torch.load(pretrained_path))
            print(f"Using model from: {pretrained_path}")

    ## DENSENET + EFFICIENTNET ##
    elif "densenet" in name or "efficientnet" in name:
        for p in model.classifier.parameters():
            p.requires_grad = True
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier.in_features, 1000),  # 1024 / 1664
            nn.ReLU(),
            nn.Linear(1000, 1)
        )

    ## REGNET ##
    elif "regnet" in name:
        for p in model.fc.parameters():
            p.requires_grad = True
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1000),  # 440 -> 1000
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        if pretrained_path != "":
            model.fc.load_state_dict(torch.load(pretrained_path))
            print(f"Using model from: {pretrained_path}")

    ## CONVNEXT ##
    elif "convnext" in name:
        for p in model.classifier[-1].parameters():
            p.requires_grad = True
        model.classifier[-1] = nn.Sequential(
            nn.Linear(model.classifier.in_features, 1000),  # 768
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        if pretrained_path != "":

    # Print the number of trainable and non-trainable parameters
    trainable, non_trainable = count_parameters(model)
    print(f"Trainable parameters: {trainable}")
    print(f"Non-trainable parameters: {non_trainable}")

    model = model.to(device)
    return model


def train(model, train_loader, val_loader, epochs=10, group="", lr=0.001, optim="adam"):
    model.train()
    device = get_device(model)

    criterion = nn.BCEWithLogitsLoss()
    if optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == "radam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Optimizer {optim} not recognized")
    
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

        ss = "" if group == "" else f"{group}/"
        wandb.log({
            f"{ss}train_loss": total_loss / len(train_loader),
            f"{ss}train_acc": train_acc,
            f"{ss}val_acc": val_acc
        }, step=epoch)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}, Val Acc: {val_acc:.2f}")

def evaluate(model, loader):
    model.eval()
    device = get_device(model)
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
    device = get_device(model)

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
            predicted_label = (torch.sigmoid(output) < 0.5).float().item()  # Apply sigmoid and threshold

        # Add image name and predicted label to results
        results.append([img_name, int(predicted_label)])

    # Write the results to a CSV file
    with open(output_csv, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["img_name", "label"])  # Write header
        writer.writerows(results)  # Write predictions

    print(f"Test predictions saved to {output_csv}")

