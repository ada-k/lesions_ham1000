import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------- CONFIGURATION ---------------------- #
DATA_DIR = "/Users/adakibet/cosmology/uel/ai_machine_vision/data"
IMAGE_DIR_1 = os.path.join(DATA_DIR, "HAM10000_images_part_1/")
IMAGE_DIR_2 = os.path.join(DATA_DIR, "HAM10000_images_part_2/")
METADATA_FILE = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
LOG_DIR = "/Users/adakibet/cosmology/uel/ai_machine_vision/runs"
MODEL_SAVE_DIR = "/Users/adakibet/cosmology/uel/ai_machine_vision/models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Define device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------- DATA LOADING ---------------------- #
def get_image_path(image_id):
    """Returns the full path of the image given its ID."""
    if os.path.exists(os.path.join(IMAGE_DIR_1, image_id + ".jpg")):
        return os.path.join(IMAGE_DIR_1, image_id + ".jpg")
    elif os.path.exists(os.path.join(IMAGE_DIR_2, image_id + ".jpg")):
        return os.path.join(IMAGE_DIR_2, image_id + ".jpg")
    return None

# Load metadata
df = pd.read_csv(METADATA_FILE)
df["image_path"] = df["image_id"].apply(get_image_path)
print(f"Missing images: {df['image_path'].isna().sum()}")

# Visualize class distribution
plt.figure(figsize=(10, 5))
sns.barplot(x=df["dx"].value_counts().index, y=df["dx"].value_counts().values, palette="viridis")
plt.title("Class Distribution in the HAM10000 Dataset")
plt.xlabel("Skin Lesion Type")
plt.ylabel("Number of Images")
plt.xticks(rotation=30)
plt.show()

# ---------------------- DATASET & TRANSFORMATIONS ---------------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class HAM10000Dataset(Dataset):
    """Custom dataset class for handling HAM10000 images."""
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["image_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_map = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
        label = label_map[self.dataframe.iloc[idx]["dx"]]
        return image, label

# Split dataset (Lesion-aware split)
train_lesions, val_lesions = train_test_split(df["lesion_id"].unique(), test_size=0.2, random_state=42)
train_df, val_df = df[df["lesion_id"].isin(train_lesions)], df[df["lesion_id"].isin(val_lesions)]
print(f"Train Set: {len(train_df)} images, Validation Set: {len(val_df)} images")

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(HAM10000Dataset(train_df, transform), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(HAM10000Dataset(val_df, transform), batch_size=batch_size, shuffle=False)

# ---------------------- MODEL TRAINING ---------------------- #
def train_model(model_name="resnet34", num_epochs=10, lr=1e-4, wd=1e-6, pretrained=True):
    """Train ResNet for skin lesion classification."""
    print(f"Training {model_name} for {num_epochs} epochs. Pretrained={pretrained}")

    # Load model
    model = models.resnet34(pretrained=pretrained) if model_name == "resnet34" else models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.to(device)

    # Compute class weights
    class_weights = torch.tensor(len(df) / (7 * df["dx"].value_counts().sort_index()), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # TensorBoard
    writer = SummaryWriter(log_dir=LOG_DIR)

    best_val_acc = 0.0
    save_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_epochs{num_epochs}_pretrained{pretrained}.pth")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training Phase
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        # Validation Phase
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total
        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        # Log metrics
        writer.add_scalar("Loss/Train", train_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/Validation", val_loss / len(val_loader), epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {time.time() - start_time:.2f}s - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

    writer.close()
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")

# Run experiment (ResNet-34, 10 epochs, pretrained)
train_model(model_name="resnet34", num_epochs=2, pretrained=True)

# ---------------------- INFERENCE ---------------------- #
def predict_image(image_path, model, device):
    """Predict the class of a single image."""
    model.eval()
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    class_map = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
    return class_map[predicted_class.item()]

# Test prediction
image_path = "data/HAM10000_images_part_1/ISIC_0027419.jpg"
print("Predicted Class:", predict_image(image_path, models.resnet34(), device))
