import torch
from torch.utils.data import Dataset

class mock_data(Dataset):
    def __init__(self, length=100, transform=None):
        """init mock data"""
        # data 
        self.data = 100*torch.ones(length,3)
        # target
        self.label = torch.ones(length,1)
        # length of data set object 
        self.length = length
        self.transform = transform
        
    def __getitem__(self,i):
        """overwrite __getitem__ method and decide what index will return"""
        training_sample = self.data[i],self.label[i]
        if self.transform:
            training_sample = self.transform(training_sample)
        return training_sample 
    
    def __len__(self):
        return self.length
