import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image


class NumberImageDataset(Dataset):
    """Create a custom subclass for number recognition training

    Args:
        Dataset (class): a subclass of Dataset for reading in custom data
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # number of images in archive
        return len(self.img_labels)

    def __getitem__(self, idx):
        """fetch record by index

        Args:
            idx (integer): the index 

        Returns:
            img [tensor]: 
            length_seq [integer]:
            seq [list digits]:
        """
        # read in img at index
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = Image.open(img_path).convert('RGB') 

        # grab annotations for index
        seq = self.img_labels.iloc[idx, 1]
        length_seq = self.img_labels.iloc[idx, 2]
        
        # transform image as we read it in 
        if self.transform:
            img = self.transform(img)

        return img, length_seq, seq
        
    

