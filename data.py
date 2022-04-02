import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, KFold

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class ImageDataset(Dataset):
    def __init__(self, path, transform, target=None, root='./data/train_images/'):
        super(ImageDataset, self).__init__()
        self.path = path
        self.target = target
        self.transform = transform
        self.root = root
        
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        target = self.target[idx]
        
        img_path = self.root + self.path[idx]
        img = cv2.imread(img_path) # np.array cv2 imread => BRG format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        return img, torch.as_tensor(target).long()
    
def get_train_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, 
                           scale_limit=0.15, 
                           rotate_limit=45, 
                           p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                # sat_shift_limit=0.2, 
                # val_shift_limit=0.2, 
                p=0.5),
        # A.RandomBrightnessContrast(
        #         brightness_limit=(-0.1, 0.1), 
        #         contrast_limit=(-0.1, 0.1), 
        #         p=0.5),
        # input img (0, 255)
        # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
        # for just min max normalize mean = 0, std = 1
        # if input mean, std (0, 255) => max_pixel_value = 1.0 
        # else input mean, std (-1, 1) default max_pixel_value = 255
        A.Normalize(mean=[0.4116, 0.4558, 0.5059], std=[0.2243, 0.2228, 0.2335]),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet for Dataset2
        ToTensorV2()
    ])

def get_valid_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.4116, 0.4558, 0.5059], std=[0.2243, 0.2228, 0.2335]),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet for Dataset2
        ToTensorV2()
    ])

def stratified_kfold(df, fold, n_split, seed=2022, input_col='image', target_col='species', just_kfold=True):
    skf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
    if just_kfold:
        for idx, (train_index, valid_index) in enumerate(skf.split(df[input_col])):
            if idx == fold:
                return train_index, valid_index
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed)
    for idx, (train_index, valid_index) in enumerate(skf.split(df[input_col], df[target_col])):
        if idx == fold:
            return train_index, valid_index

        
