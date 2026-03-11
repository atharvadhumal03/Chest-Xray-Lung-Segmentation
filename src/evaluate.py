# =========================================================================================================================================
# Imports
# =========================================================================================================================================
import sys
sys.path.append('/src')

import os
import torch
import random
from glob import glob
from tqdm import tqdm
from model import UNet
import matplotlib.pyplot as plt
from dataset import split_datasets
from torch.utils.data import DataLoader
from metrics import dice_loss, iou_score


# =========================================================================================================================================
# Paths
# =========================================================================================================================================
DATA_DIR = 'Chest-X-Ray'
OUTPUT_DIR = 'outputs'
MODEL_PATH = 'outputs/checkpoints/best_model.pth'

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# =========================================================================================================================================
# Loading Data
# =========================================================================================================================================
image_paths = sorted(glob(os.path.join(DATA_DIR, 'image', '[0-9]*')))
mask_paths = sorted(glob(os.path.join(DATA_DIR, 'mask', '[0-9]*')))

_, _, test_dataset = split_datasets(image_paths, mask_paths)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# =========================================================================================================================================
# Loading Model
# =========================================================================================================================================
print("Loading model...")

model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully!")
print(f"Using device: {DEVICE}")

# =========================================================================================================================================
# Model Eval
# =========================================================================================================================================
def evaluate(model, loader, criterion, metric, device):
    model.eval()
    running_dice = 0.0
    running_IoU = 0.0

    pbar = tqdm(loader, desc='Validation')

    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, masks)
            IoU_score = metric(outputs, masks)
            running_dice += loss.item()
            running_IoU += IoU_score.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'IoU': f'{IoU_score.item():.4f}'})
    
    return 1 - running_dice / len(loader), running_IoU / len(loader)

# =========================================================================================================================================
# Visualization
# =========================================================================================================================================

def visualize(model, dataset, device, num_samples=5):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    axes[0, 0].set_title('X-Ray', fontsize=14)
    axes[0, 1].set_title('Ground Truth', fontsize=14)
    axes[0, 2].set_title('Prediction', fontsize=14)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            
            # Run inference
            output = model(image.unsqueeze(0).to(device))
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred = (pred > 0.5).astype(float)
            
            # Convert image and mask for plotting
            img = image.squeeze().cpu().numpy()
            msk = mask.squeeze().cpu().numpy()
            
            # Plot
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 1].imshow(msk, cmap='gray')
            axes[i, 2].imshow(pred, cmap='gray')
            
            for j in range(3):
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/predictions/sample_predictions.png', dpi=150)
    plt.show()
    print("Saved to outputs/predictions/sample_predictions.png")

# =========================================================================================================================================
# Main Function
# =========================================================================================================================================
if __name__ == "__main__":
    print("Evaluating on test set...")
    dice_score, iou = evaluate(model, test_loader, dice_loss, iou_score, DEVICE)
    print(f"Test Dice Score: {dice_score:.4f}")
    print(f"Test IoU: {iou:.4f}")
    
    print("=" * 50)
    print("Evaluation Complete!")
    print("=" * 50)

    print("Visualizing..")
    visualize(model, test_dataset, DEVICE)