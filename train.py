
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from ultralytics import YOLO
from albumentations import Compose, HorizontalFlip, RandomCrop, Normalize
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Step 2: Define Dataset Class
class DroneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        
        mask = torch.tensor(mask, dtype=torch.long)  # Convert mask to long tensor
        mask = torch.clamp(mask, min=0, max=19)  # Ensure mask values are within range
        
        return image, mask

# Define transformations
transform = Compose([
    HorizontalFlip(p=0.5),
    RandomCrop(512, 512),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Dataset paths
image_dir = "/home/atharva/Documents/Semantic Segmentation (Drone)/Dataset/dataset/semantic_drone_dataset/original_images/"
mask_dir = "/home/atharva/Documents/Semantic Segmentation (Drone)/Dataset/dataset/semantic_drone_dataset/label_images_semantic/"

# Load dataset
dataset = DroneDataset(image_dir, mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Step 3: Load Pretrained DeepLabV3+ Model
model = deeplabv3_resnet50(pretrained=True)
num_classes = 20  # Matching dataset classes
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
model = model.cuda()

# Load saved model instead of training
model.load_state_dict(torch.load("deeplabv3_drone.pth"))
model.eval()

# Step 4: Evaluate Model Performance
def evaluate_model(model, dataloader):
    model.eval()
    total_iou = 0
    total_acc = 0
    num_batches = 0
    iou_list = []
    acc_list = []
    class_iou = np.zeros(num_classes)
    class_count = np.zeros(num_classes)
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)
            
            for c in range(num_classes):
                intersection = ((preds == c) & (masks == c)).float().sum()
                union = ((preds == c) | (masks == c)).float().sum()
                if union > 0:
                    class_iou[c] += (intersection / union).item()
                    class_count[c] += 1
            
            intersection = (preds & masks).float().sum((1,2))
            union = (preds | masks).float().sum((1,2))
            iou = (intersection + 1e-6) / (union + 1e-6)
            acc = (preds == masks).float().mean()
            
            iou_list.append(iou.mean().item())
            acc_list.append(acc.item())
            total_iou += iou.mean().item()
            total_acc += acc.item()
            num_batches += 1
    
    class_iou = class_iou / np.maximum(class_count, 1)
    
    print(f"Mean IoU: {total_iou / num_batches:.4f}")
    print(f"Pixel Accuracy: {total_acc / num_batches:.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(iou_list, label='IoU')
    plt.plot(acc_list, label='Pixel Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('IoU & Accuracy Trend')
    plt.show()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(range(num_classes)), y=class_iou)
    plt.xlabel('Class')
    plt.ylabel('IoU')
    plt.title('Per-Class IoU')
    plt.show()

evaluate_model(model, dataloader)

# Step 5: Visualize Predictions
def visualize_predictions(model, dataloader):
    model.eval()
    images, masks = next(iter(dataloader))
    images, masks = images.cuda(), masks.cuda()
    with torch.no_grad():
        outputs = model(images)['out']
        preds = torch.argmax(outputs, dim=1)
    
    fig, axes = plt.subplots(3, len(images), figsize=(15, 6))
    for i in range(len(images)):
        axes[0, i].imshow(images[i].permute(1,2,0).cpu().numpy())
        axes[0, i].set_title("Image")
        axes[1, i].imshow(masks[i].cpu().numpy(), cmap='gray')
        axes[1, i].set_title("Ground Truth")
        axes[2, i].imshow(preds[i].cpu().numpy(), cmap='gray')
        axes[2, i].set_title("Prediction")
    plt.show()

visualize_predictions(model, dataloader)
print("Evaluation & Visualization Completed!")
