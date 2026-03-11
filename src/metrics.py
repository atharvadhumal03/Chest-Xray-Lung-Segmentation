import torch

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
# Metric - IoU Score
# =========================================================================================================================================
def iou_score(predictions, targets, smooth=1e-6):
    predictions = torch.sigmoid(predictions)
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)