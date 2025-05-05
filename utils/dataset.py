import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class MineDataset(Dataset):
    def init(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        for class_name in ['mine', 'no_mine']:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    label = 1 if class_name == 'mine' else 0
                    samples.append((img_path, label))
        return samples
    
    def len(self):
        return len(self.samples)
    
    def getitem(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)