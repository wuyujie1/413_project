
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, dist_map_path=None):

        self.images_path = images_path
        self.masks_path = masks_path
        self.dist_map_path = dist_map_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (128,256))
        image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (128,256))
        mask = mask/255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        if self.dist_map_path is None:
            return image, mask, None
        else:
            """Reading Dist_Map for the Boundary Loss"""
            map = cv2.imread(self.dist_map_path[index], cv2.IMREAD_GRAYSCALE)
            map = cv2.resize(map, (128, 256))
            # map = map / 255.0  ## (512, 512)
            map = np.expand_dims(map, axis=0)  ## (1, 512, 512)
            map = map.astype(np.float32)
            map = torch.from_numpy(map)

            return image, mask, map

    def __len__(self):
        return self.n_samples
