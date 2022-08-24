import torch
from torch.utils.data import Dataset

class TransformSample(object):
    def __init__(self, scale_data=10, scale_label=5):
        self.scale_data = scale_data
        self.scale_label = scale_label
    def __call__(self,sample):
        data = sample[0] * self.scale_data
        label = sample[1] * self.scale_label
        return (data,label)
        

class MockData(Dataset):
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
    
    
def main():
    transform = TransformSample()
    mock_data = MockData(100, transform = transform)
    print(mock_data[1]) 
    

if __name__ == "__main__":
    main()
    