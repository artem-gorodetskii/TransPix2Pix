import torch
from torch.utils.data import Dataset
import albumentations as A
from skimage import io, transform


class CatsDataset(Dataset):
    """This class loads image from dataset and performs image preprocessing.
    Subclass Dataset class from torch.utils.data.
    """
    def __init__(self, datapath, dataframe, augment=None):
        """
        Initialize CatsDataset.
        
        :datapath: str, path to the dataset
        :dataframe: DataFrame, includes embeddings from Style-Pre-Net
        :augment: bool, if True, augment data
        """
        self.datapath = datapath
        self.dataframe = dataframe
        self.augment = augment


    def __len__(self):
        return len(self.datapath)


    def __getitem__(self, idx):
        image_path = self.datapath[idx]
        row = self.dataframe[self.dataframe.path == f'{image_path}']
        embedding = row['embedding'].values[0]
        image = io.imread(image_path)
        
        # split sketch and image
        w = image.shape[1]  // 2
        sketch = image[:, :w, :]
        image = image[:, w:, :]
        
        # normalize data
        sketch = ( sketch / 127.5)  - 1
        image = ( image / 127.5)  - 1
        
        # augment data
        if self.augment:
            aug = A.OneOf([A.HorizontalFlip(p=1), A.RandomSizedCrop(min_max_height=(230, 230), height=256, width=256, p=1)], p=0.5)
            augmented = aug(image=image, mask=sketch)
            image = augmented['image']
            sketch = augmented['mask']

        image = torch.FloatTensor(image)
        sketch = torch.FloatTensor(sketch)
        embedding = torch.FloatTensor(embedding)
        
        # check image dimension. If [height, width], add channel dimmension. Permute dimension to [channel, height, width]
        if len(image.shape) == 2:
            image = torch.stack([image] * 3)
        else:
            image = image.permute(2, 0, 1)

        if len(sketch.shape) == 2:
            sketch = torch.stack([sketch] * 3)
        else:
            sketch = sketch.permute(2, 0, 1)

        return sketch, image, embedding
        