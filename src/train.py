# =========================================================================================================================================
# Imports
# =========================================================================================================================================

import os
import sys
sys.path.append('../src')

import torch
from tqdm import tqdm
from glob import glob
from model import UNet
import torch.optim as optim
from dataset import split_datasets
from torch.utils.data import DataLoader

# =========================================================================================================================================
# Loss Function - Dice Loss
# =========================================================================================================================================
def dice_loss(predictions, targets, smooth=1e-6):
    # Apply sigmoid to convert logits to 0-1
    predictions = torch.sigmoid(predictions)
    
    # Flatten both tensors
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Compute intersection and dice score
    intersection = (predictions * targets).sum()
    dice_score = (2 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    
    return 1 - dice_score

# =========================================================================================================================================
# Intialize model
# =========================================================================================================================================
model = UNet(in_channels=1, out_channels=1)

# =========================================================================================================================================
# Hyperparameters
# =========================================================================================================================================
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 1
DEVICE = None

# =========================================================================================================================================
# Intializing set-up
# =========================================================================================================================================
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

print(f"Using device: {DEVICE}\n")
model = model.to(DEVICE)

# =========================================================================================================================================
# Loading data
# =========================================================================================================================================
DATA_PATH_IMG = "../Chest-X-Ray/image"
DATA_PATH_MSK = "../Chest-X-Ray/mask"

image_paths = sorted(glob(os.path.join(DATA_PATH_IMG, '[0-9]*')))
mask_paths = sorted(glob(os.path.join(DATA_PATH_MSK, '[0-9]*')))

train_dataset, val_dataset, test_dataset = split_datasets(image_paths, mask_paths)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================================================================================================================================
# Train function
# =========================================================================================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_dice = 0.0

    pbar = tqdm(loader, desc='Training')

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        running_dice += loss.item()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'dice loss': f'{loss.item():.4f}'})

    return running_dice / len(loader)

# =========================================================================================================================================
# Validate/Evaluate function
# =========================================================================================================================================
def validate(model, loader, criterion, device):
    model.eval()
    running_dice = 0.0

    pbar = tqdm(loader, desc='Validation')

    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_dice += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_dice / len(loader)

# =========================================================================================================================================
# Model Training
# =========================================================================================================================================
if __name__ == "__main__":
    train_dices, val_dices = [], []
    best_val_dice = float('inf')

    for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            
            # Train
            train_dice = train_one_epoch(model, train_loader, dice_loss, optimizer, DEVICE)
            train_dices.append(train_dice)
            
            # Validate
            val_dice = validate(model, val_loader, dice_loss, DEVICE)
            val_dices.append(val_dice)
            
            print(f"Train Dice: {train_dice:.4f}")
            print(f"Val Dice: {val_dice:.4f}")
            
            # Save best model
            if val_dice < best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), '../outputs/checkpoints/best_model.pth')
                print(f"Saved best model (Dice: {val_dice:.4f})")

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                }, f'../outputs/checkpoints/checkpoint_epoch{epoch+1}.pth')
                print(f"Saved checkpoint at epoch {epoch+1}")
        
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best Val Dice Score: {best_val_dice:.4f}")
    print("="*50)