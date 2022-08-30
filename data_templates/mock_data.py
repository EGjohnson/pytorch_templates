import torch
from torch.utils.data import Dataset


torch.manual_seed(1)

# write your transforms
class ScaleSample(object):
    def __init__(self, scale_data=10.0, scale_label=5.0):
        """constructor: transformation obj with attributes for your transform"""
        self.scale_data = scale_data
        self.scale_label = scale_label
    def __call__(self,sample):
        """executer: how you want to transform your raw data"""
        data = sample[0] * self.scale_data
        label = sample[1] * self.scale_label
        return (data,label)

 
# class EncodeLabel(object):
#     def __init__(self, high = ):
#         """construct transformation with attributes for your transform"""
#         self.scale_data = scale_data
#         self.scale_label = scale_label
#     def __call__(self,sample):
#         """write here how you want to transform your raw data"""
#         data = sample[0] * self.scale_data
#         label = sample[1] * self.scale_label
#         return (data,label)



# CREATING A MOCK DATA SET OBJECT       
# Need to subclass Dataset and customize the three below methods for dataset object:
# 1. __init__: constructor
# 2. __getitem__: method to get piece of data
# 3. __len__: method to return length of data set

class MockData(Dataset):
    def __init__(self, length=200, transform=None):
        """construct mock data object"""
        # mock data 
        self.data = 100*torch.rand(length,3) #x,y,z coordinate
        # mock target label
        self.label = torch.rand(length,1) # temperature
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
    # 1. create transform for raw data
    scale_transform = ScaleSample(300, 0.5)
    # 2. create mock Dataset object
    mock_data = MockData(24, transform = scale_transform)
    # 3. print outputs
    print(f'the length of our mock data is {len(mock_data)}')
    i = 12
    print("Apply scale transform")
    print(f'at index {i} the data  x,y,z coordinate is {mock_data[i][0]} and the temperature label is {mock_data[i][1]}')
    print("Apply both transforms")
    

if __name__ == "__main__":
    main()
    