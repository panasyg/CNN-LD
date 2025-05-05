import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import numpy as np
import torch
from torch.utils.data import Dataset

def _extract_array(obj):
    if isinstance(obj, np.ndarray) and obj.dtype==object:
        return _extract_array(obj.item())
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            try: return _extract_array(v)
            except: continue
    if isinstance(obj, (list,tuple)):
        for v in obj:
            try: return _extract_array(v)
            except: continue
    raise ValueError("No ndarray in raw")

class MineDataset(Dataset):
    def init(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.samples = list(self.img_dir.glob('**/*.png'))
        self.transform = transform or self._default_transform()
        
    def _default_transform(self):
        return A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])
    
    def getitem(self, idx):
        img_path = self.samples[idx]
        img = cv2.imread(str(img_path))[:,:,::-1]  # BGR to RGB
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        label = 1 if 'mine' in img_path.parent.name else 0
        return torch.FloatTensor(img).permute(2,0,1), torch.tensor(label)
