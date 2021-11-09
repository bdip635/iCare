import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import pandas as pd
import  albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class RFMiD_Dataset(Dataset):

  def __init__(self, main_dir, csv_path, image_shape, transforms=None):
    super(RFMiD_Dataset, self).__init__()
    self.main_dir= main_dir
    self.df= pd.read_csv(csv_path)
    self.image_shape= image_shape
    self.transforms= transforms
  
  def __len__(self):
    return len(self.df)

  def get_class_weights(self):
    labels= self.df['DR'].to_numpy()
    return {class_id: (1 - np.sum(labels==class_id)/len(labels)) for class_id in np.unique(labels)} 
  
  def __getitem__(self, index):
    id= self.df.iloc[index]['ID']
    image= cv2.resize(cv2.imread(os.path.join(self.main_dir, str(id)+'.png')), self.image_shape[1:], interpolation= cv2.INTER_CUBIC)[:,:,::-1]

    if self.transforms is not None:
      image= self.transforms(image=image)['image']/255.

    return image, self.df.iloc[index]['DR']
    
def plotter(loader, figsize):
  images, labels= next(iter(loader))

  plt.figure(figsize=figsize)
  for r in range(loader.batch_size):
      plt.subplot(4,8,r+1)
      f= plt.imshow(images[r].permute(2,1,0))
      plt.title(labels[r].item())
  plt.show()