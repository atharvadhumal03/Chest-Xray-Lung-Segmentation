# =========================================================================================================================================
# Imports
# =========================================================================================================================================

import os
import sys
sys.path.append('../src')

import torch
import argparse
from tqdm import tqdm
from glob import glob
from model import UNet
import torch.optim as optim
from metrics import dice_loss
from dataset import split_datasets
from torch.utils.data import DataLoader

# =========================================================================================================================================
# argparse - handling file paths
# =========================================================================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../Chest-X-Ray')
parser.add_argument('--output_dir', type=str, default='../outputs')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()

# =========================================================================================================================================
# Hyperparameters
# =========================================================================================================================================
LEARNING_RATE = 1e-4
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
DEVICE = None

# =========================================================================================================================================
# Intializing set-up
# =========================================================================================================================================
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

print(f"Using device: {DEVICE}\n")

model = UNet(in_channels=1, out_channels=1)
model = model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================================================================================================================================
# Loading data
# =========================================================================================================================================
DATA_PATH_IMG = os.path.join(args.data_dir, 'image')
DATA_PATH_MSK = os.path.join(args.data_dir, 'mask')

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
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoints', 'best_model.pth'))
                print(f"Saved best model (Dice: {val_dice:.4f})")

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                }, os.path.join(args.output_dir, 'checkpoints', f'checkpoint_epoch{epoch+1}.pth'))
                print(f"Saved checkpoint at epoch {epoch+1}")
        
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best Val Dice Score: {best_val_dice:.4f}")
    print("="*50)