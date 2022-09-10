import torch
from torch.utils.data import Dataset
import albumentations as A
from skimage import io, transform
import cv2

class CatsDataset(Dataset):
    """
    StyleEncoder data generator.
    """
    def __init__(self, imagespath, augment=None):
        self.imagespath = imagespath
        self.augment = augment
        self.names = ['/black/', '/blackwhite/', '/gray/', '/siberian/', '/siamese/', '/ginger/', '/gingerwhite/', '/white/', '/other/']

    def __len__(self):
        return len(self.imagespath)

    def __getitem__(self, idx):
        image_path = self.imagespath[idx]
        image = io.imread(image_path)

        # Normalize data
        image = transform.resize(image, (64, 64)) 
        image = image.astype('float32')
        
        # Augment data
        if self.augment:
            aug = A.OneOf([A.HorizontalFlip(p=1), 
                           A.RandomSizedCrop(min_max_height=(54, 64), height=64, width=64, p=1),
                           A.Blur(p=1),
                           A.GlassBlur(sigma=0.1, max_delta=2, p=1),
                           A.geometric.rotate.Rotate(limit=20, border_mode=cv2.BORDER_REPLICATE, p=1),
                           A.ShiftScaleRotate(shift_limit = 0.1, scale_limit =0, rotate_limit=0, interpolation=1, border_mode=cv2.BORDER_REPLICATE, p=1),
                           A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1)
                          ], p=0.5)
            augmented = aug(image=image)
            image = augmented['image']

        image = torch.FloatTensor(image)
        image = image.permute(2, 0, 1)
        
        # Assign a label to the image by the folder name
        for name in self.names:
            if name in image_path:
                label_idx = torch.tensor(self.names.index(name), dtype=torch.long)
                
        return image, label_idx, image_path
