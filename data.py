import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# Define image transformations
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to fit model input
    transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
    transforms.RandomRotation(20),  # Random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to fit model input
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])


def load_data(full_dataset=False):
    if full_dataset:
        from torch.utils.data import ConcatDataset
        train_train_dataset = ImageFolder(root="dataset/train", transform=train_transform)
        val_train_dataset = ImageFolder(root="dataset/val", transform=train_transform)
        train_dataset = ConcatDataset([train_train_dataset, val_train_dataset])
    else:
        train_dataset = ImageFolder(root="dataset/train", transform=train_transform)
    val_dataset = ImageFolder(root="dataset/val", transform=transform)
    return train_dataset, val_dataset


def create_data_loaders(train_dataset, val_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader