

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
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

# Check sample data
sample_img, sample_mask = next(iter(dataloader))
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(sample_img[0].permute(1,2,0))
plt.title("Sample Image")
plt.subplot(1,2,2)
plt.imshow(sample_mask[0], cmap='gray')
plt.title("Sample Mask")
plt.show()
