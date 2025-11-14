import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from src.settings import dataset_config as config


class ForgeryDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, is_authentic=None):
        self.IMG_SIZE = config.get('image_size', 256)

        self.image_paths = image_paths
        self.mask_paths = mask_paths if mask_paths is not None else [
            None] * len(image_paths)
        self.is_authentic = is_authentic if is_authentic is not None else [
            False] * len(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE)) / 255.0
        img = torch.from_numpy(img).float().permute(2, 0, 1)

        # Load or create mask
        if self.is_authentic[idx]:
            mask = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)
        else:
            mask = np.load(self.mask_paths[idx])
            if mask.ndim == 3:
                mask = mask[0, :, :]
            mask = cv2.resize(mask.astype(np.uint8),
                              (self.IMG_SIZE, self.IMG_SIZE))
            mask = (mask > 0).astype(np.float32)

        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return img, mask
