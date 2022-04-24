import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def stratified_kfold(df, fold, n_split, seed=2022, input_col='image', target_col='species'):
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed)
    for idx, (train_index, valid_index) in enumerate(skf.split(df[input_col], df[target_col])):
        if idx == fold:
            return train_index, valid_index
        
        
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class ImageDataset(Dataset):
    def __init__(self, path, transform, target=None, root='../data/train_images/'):
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
        img = load_image(img_path)
        img = self.transform(image=img)["image"]

        return img, torch.as_tensor(target).long()

        
def get_train_transforms(image_size):
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, 
                           scale_limit=0.15, 
                           rotate_limit=30, 
                           p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2,
                val_shift_limit=0.2, 
                p=0.5),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), 
                contrast_limit=(-0.1, 0.1),
                p=0.5),
        A.GaussNoise(var_limit=5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ToTensorV2()
    ])


def get_valid_transforms(image_size):
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])