import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class GOT10kDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sequences = [os.path.join(root_dir, seq) for seq in os.listdir(root_dir) if not seq.startswith('.')]
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        while True:  # Keep trying until we get valid data
            try:
                seq_path = self.sequences[idx]
                frames = sorted([f for f in os.listdir(seq_path) 
                            if f.endswith('.jpg') and not f.startswith('.')])
                if len(frames) < 2:
                    idx = (idx + 1) % len(self)
                    continue

                template = Image.open(os.path.join(seq_path, frames[0])).convert('RGB')
                search = Image.open(os.path.join(seq_path, frames[1])).convert('RGB')
                width, height = template.size

                with open(os.path.join(seq_path, 'groundtruth.txt'), 'r') as f:
                    bbox = f.readline().strip().split(',')
                    bbox = list(map(float, bbox))  # [x1, y1, x2, y2]
                    bbox = np.array(bbox) / [width, height, width, height]  # normalize to [0, 1]

                return {
                    'template': self.transform(template),
                    'search': self.transform(search),
                    'label': torch.tensor(bbox, dtype=torch.float32)
                }
            except Exception as e:
                print(f"Error loading sequence {seq_path}: {e}")
                idx = (idx + 1) % len(self)

    def crop_patch(img, bbox, scale=1.0):
        x, y, w, h = bbox
        crop = img[y:y+h, x:x+w]
        # Optionally apply scaling to the crop
        if scale != 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale)
        return crop
