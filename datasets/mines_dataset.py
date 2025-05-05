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

class MinesDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.transform = transform
        data_dir = os.path.join(os.path.dirname(__file__), 'giuriati_2')
        files = sorted(f for f in os.listdir(data_dir) if f.endswith('.npy'))
        vols = []
        for fn in files:
            raw = np.load(os.path.join(data_dir, fn), allow_pickle=True)
            arr = _extract_array(raw).astype(np.float32)
            vols.append(arr)
        self.vols = vols
        # Flatten all slices into list of (vol_idx, slice_idx)
        self.index = []
        for vid, vol in enumerate(self.vols):
            D = vol.shape[0]
            for z in range(D):
                self.index.append((vid,z))
        # Optionally split train/val here by slicing index list

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        vid,z = self.index[idx]
        slice2d = self.vols[vid][z]             # shape (H,W)
        img = torch.from_numpy(slice2d).unsqueeze(0)  # (1,H,W)
        mask = (slice2d>slice2d.mean()).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        if self.transform:
            img,mask = self.transform(img,mask)
        return img,mask
