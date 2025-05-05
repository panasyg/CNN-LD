import torch
import numpy as np
from torch.utils.data import Dataset
import os

class MinesDataset(Dataset):
    def __init__(self, split='train', transform=None):
        """
        Датасет для завантаження зображень і міток мін.
        :param split: 'train' або 'val' для вибору підмножини.
        :param transform: додаткові перетворення.
        """
        self.split = split
        self.transform = transform
        self.data_dir = './datasets/giuriati_2'
        self.img_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npy')])

        # Завантаження зображень і міток
        self.images = []
        self.masks = []

        for img_file in self.img_files:
            # Завантажуємо зображення
            img_path = os.path.join(self.data_dir, img_file)
            img_data = np.load(img_path)

            # Припустимо, що маска — це зображення з деяким порогом
            mask_data = (img_data > np.mean(img_data)).astype(np.float32)  # це просто приклад, як можна створити маску

            self.images.append(img_data)
            self.masks.append(mask_data)

        self.images = np.array(self.images)
        self.masks = np.array(self.masks)

        # Розбиття на train/val
        num_samples = len(self.images)
        if self.split == 'train':
            self.images = self.images[:int(0.8 * num_samples)]
            self.masks = self.masks[:int(0.8 * num_samples)]
        elif self.split == 'val':
            self.images = self.images[int(0.8 * num_samples):]
            self.masks = self.masks[int(0.8 * num_samples):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Повертає зображення і маску для конкретного індексу
        :param idx: індекс елемента
        :return: (image, mask)
        """
        img = self.images[idx]
        mask = self.masks[idx]

        # Перетворення в тензори PyTorch
        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        # Додавання виміру каналу (B, C, H, W)
        img = img.unsqueeze(0)  # (1, H, W)
        mask = mask.unsqueeze(0)  # (1, H, W)

        # Якщо є перетворення (аугментація), застосовуємо
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

