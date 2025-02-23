import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Load labels
data_path = "dataset/train_images"
labels_file = "dataset/train_data.csv"
labels = dict(pd.read_csv(labels_file).values)

# Define output directories
train_dir = "dataset/train"
val_dir = "dataset/val"
malaria_dir_train = os.path.join(train_dir, "malaria")
no_malaria_dir_train = os.path.join(train_dir, "no_malaria")
malaria_dir_val = os.path.join(val_dir, "malaria")
no_malaria_dir_val = os.path.join(val_dir, "no_malaria")

# Create necessary directories
for folder in [malaria_dir_train, no_malaria_dir_train, malaria_dir_val, no_malaria_dir_val]:
    os.makedirs(folder, exist_ok=True)

# Split data into train and validation sets
filenames = list(labels.keys())
train_files, val_files = train_test_split(filenames, test_size=0.2, random_state=42)

# Function to move files
def move_files(file_list, target_dir):
    for file in file_list:
        src_path = os.path.join(data_path, file)
        if os.path.exists(src_path):
            label = labels[file]
            dest_folder = malaria_dir_train if label == 1 else no_malaria_dir_train
            if target_dir == "val":
                dest_folder = malaria_dir_val if label == 1 else no_malaria_dir_val
            shutil.move(src_path, os.path.join(dest_folder, file))

# Move files to respective directories
move_files(train_files, "train")
move_files(val_files, "val")

print("✅ Dataset preprocessing completed! Images have been organized into train and val folders.")
