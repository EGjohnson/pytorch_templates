import torch
from torch.utils.data import Dataset

class transform_sample(self,addx=1,multy=1):
    def __init__(self, scale_data=10, scale_label=5):
        self.scale_data = scale_data
        self.scale_label = scale_label
    def __call__(self,sample):
        data = sample[0] * scale_data
        label = sample[1] * scale_label
        

class mock_data(Dataset):
    def __init__(self, length=100, transform=None):
        """init mock data"""
        # data 
        self.data = 100*torch.ones(length,3)
        # target label
        self.label = torch.ones(length,1)
        # length of data set object 
        self.length = length
        self.transform = transform
        
    def __getitem__(self,i):
        """write __getitem__ method so will return samples by index"""
        training_sample = self.data[i],self.label[i]
        if self.transform:
            training_sample = self.transform(training_sample)
        return training_sample 
    
    def __len__(self):
        return self.length
