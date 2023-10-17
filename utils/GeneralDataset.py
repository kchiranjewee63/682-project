import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations import Compose
import os
import torchio as tio
import torch
import random
import math
from PIL import Image

class GeneralDataset(Dataset):
    def __init__(self, images, masks, center_crop_size):
        self.images = images
        self.masks = masks
        self.center_crop_size = center_crop_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        
        stacked_images = []
      
        for modality in image_path:
            img = self.load_img(modality)

            img = self.crop_or_pad_img(img)

            img = self.normalize(img)
            stacked_images.append(img)
    
        stacked_images = np.stack(stacked_images)
        
        mask = self.load_img(mask_path)
        mask = self.crop_or_pad_img(mask)
        mask = mask[None, ...]

        return {
            "image": stacked_images,
            "mask": mask,
            "image_path": image_path,
            "mask_path": mask_path
        }

    def load_img(self, file_path):
        data = Image.open(file_path).convert('L')
        data = np.array(data)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min + 1e-9)
    
    def crop_or_pad_img(self, image):
        target_height, target_width = self.center_crop_size
        height, width = image.shape

        if height > target_height:
            starty = height//2 - target_height//2
            image = image[starty:starty+target_height, :]

        if width > target_width:
            startx = width//2 - target_width//2
            image = image[:, startx:startx+target_width]

        height, width = image.shape
        pad_height = target_height - height
        pad_width = target_width - width
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

        return image