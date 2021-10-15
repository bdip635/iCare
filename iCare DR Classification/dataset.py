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
from config import *


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



def get_dataloaders():
  
  train_dir= 'data/Training Set'
  val_dir= 'data/Validation Set'
  test_dir= 'data/Testing Set'
  train_csv_path= 'data/a. RFMiD_Training_Labels.csv'
  val_csv_path= 'data/b. RFMiD_Validation_Labels.csv'
  test_csv_path= 'data/c. RFMiD_Testing_Labels.csv'
  check_pt_file= 'weights/wide_resnet50_2.pth.tar'
  input_shape= (3, )+IMG_SIZE
  transforms = A.Compose(
      [A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5), 
        A.CLAHE(p=1),
        A.RandomBrightnessContrast(p=0.8),
        A.RandomGamma(p=0.8),
        ToTensorV2()])

  train_dataset= RFMiD_Dataset(train_dir, train_csv_path, input_shape, transforms)
  val_dataset= RFMiD_Dataset(val_dir, val_csv_path, input_shape, transforms=ToTensorV2())
  test_dataset= RFMiD_Dataset(test_dir, test_csv_path, input_shape, transforms=ToTensorV2())

  train_loader= DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
  val_loader= DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
  test_loader= DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

  return train_loader, val_loader, test_loader

def plotter(loader, figsize):
  images, labels= next(iter(loader))

  plt.figure(figsize=figsize)
  for r in range(loader.batch_size):
      plt.subplot(4,8,r+1)
      f= plt.imshow(images[r].permute(2,1,0))
      plt.title(labels[r].item())
  plt.show()
  
if __name__=='__main__':
    print("No Output!!")