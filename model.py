import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: [B, 384, 1, 1]
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 256), nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 4)  # Output: [x1, y1, x2, y2]
        )

    def forward(self, z, x):
        z_feat = self.feature_extractor(z)
        x_feat = self.feature_extractor(x)
        combined = z_feat * x_feat  # Cross-correlation
        pooled = self.pool(combined)
        out = self.regressor(pooled)
        return torch.sigmoid(out)  # Ensure outputs are in [0, 1]






def iou_loss(pred_boxes, true_boxes):
    """
    Calculate IoU loss for batches of boxes.
    Args:
        pred_boxes: [B, 4] tensor of predicted boxes (x1, y1, x2, y2)
        true_boxes: [B, 4] tensor of ground truth boxes (x1, y1, x2, y2)
    Returns:
        IoU loss for each box pair in the batch
    """
    # Calculate intersection coordinates
    xA = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
    yA = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
    xB = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
    yB = torch.min(pred_boxes[:, 3], true_boxes[:, 3])

    # Calculate areas
    interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    boxAArea = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    boxBArea = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])

    # Calculate IoU
    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)  # Add small epsilon to avoid division by zero
    return 1 - iou  # IoU loss (we minimize 1 - IoU) # IoU loss (we minimize 1 - IoU)

