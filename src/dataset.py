import os
import numpy as np
from glob import glob
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# Dataset Class
class LungSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transformations=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transformations = transformations

        print(f"Found {len(self.image_paths)} image-mask pairs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.mask_paths[index])

        image = image.convert("L")
        mask = mask.convert("L")

        image = np.array(image)
        mask = np.array(mask)

        mask = (mask > 127).astype(np.float32) # type: ignore

        if self.transformations:
            transformed = self.transformations(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask

# Transformations
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=0.5),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

# Train Test Val split
def split_datasets(image_paths, mask_paths):
    train_val_imgs, test_imgs, train_val_msks, test_msks = train_test_split(
        image_paths, mask_paths, test_size=0.15, random_state=42
    )
    train_imgs, val_imgs, train_msks, val_msks = train_test_split(
        train_val_imgs, train_val_msks, test_size=0.176, random_state=42
    )
    
    train_dataset = LungSegDataset(train_imgs, train_msks, transformations=train_transform)
    val_dataset = LungSegDataset(val_imgs, val_msks, transformations=val_transform)
    test_dataset = LungSegDataset(test_imgs, test_msks, transformations=val_transform)
    
    return train_dataset, val_dataset, test_dataset