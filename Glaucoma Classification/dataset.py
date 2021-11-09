import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset, random_split

import os
import numpy as np
import math
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class RED_RCNN_Dataset(Dataset):
  def __init__(self, root, transforms, csv_path, img_shape=(512,512)):
    self.root = root
    self.transforms = transforms
    self.img_shape = img_shape
    self.df = pd.read_csv(csv_path)
    self.classes = np.array([255, 128, 0]) #{0:255, 1:128, 2:0}
  
  def __len__(self):
    return len(self.df)
  
  def process_mask_to_onehot(self, mask):
    obj_ids = self.classes # np.unique(mask)
    obj_ids = obj_ids[1:]
    onehot = (mask<=obj_ids[:, None, None])
    return np.transpose(onehot, (1,2,0)) #(512,512,2)
  
  def get_bbox(self, onehot):
    boxes = []
    for i in range(len(self.classes[1:])):
      pos = np.where(onehot[:,:,i]) # Return the indices of the elements that are non-zero.
      xmin = np.min(pos[1])
      xmax = np.max(pos[1])
      ymin = np.min(pos[0])
      ymax = np.max(pos[0])
      boxes.append([xmin, ymin, xmax, ymax])
    return np.array(boxes)
      
  def __getitem__(self, idx):
    img_path = os.path.join(self.root, self.df.iloc[idx, 1])
    mask_path = os.path.join(self.root, self.df.iloc[idx, 2])
    label = int(self.df.iloc[idx, 3])

    img = cv2.imread(img_path)
    img = cv2.cvtColor(cv2.resize(img, self.img_shape, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    mask = cv2.resize(cv2.imread(mask_path, 0), self.img_shape, interpolation=cv2.INTER_NEAREST)
    
    onehot = self.process_mask_to_onehot(mask)
    boxes = self.get_bbox(onehot)

    num_objs = len(self.classes[1:])

    target = {}
    target['boxes'] = boxes
    target['labels'] = [1,2] #Disc-1, Cup-2
    target['masks'] = onehot
    target['image_id'] = torch.tensor([idx])
    target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64) #
    
    if self.transforms is not None:
      transformed = self.transforms(image=img, mask=target['masks'], bboxes=target['boxes'], bbox_classes=target['labels'])
      
      img = transformed['image']
      target['masks'] = transformed['mask'].permute(2, 0, 1) 

      target['boxes'] = np.array(transformed['bboxes'], dtype=int)
      # target['boxes'] = np.array(transformed['bboxes'])
      # for i in target['boxes']:
      #   i[1] = i[1]*self.img_shape[0]/3
      #   i[3] = i[3]*self.img_shape[0]/3
      # target['boxes'] = np.array(target['boxes'], dtype=int)
      target['labels'] = np.array(transformed['bbox_classes'], dtype=int)
      boxes_t = target['boxes']
      target['area'] = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])

    return img, target

def visualize_bbox(img, box, label, color=[(0,255,0),(0,0,255)], text_color=(0,0,0), thickness=2):
    x_min, y_min, x_max, y_max = box
    if label == 1:
      label_name = 'Disc'
      color = color[0]
    elif label == 2:
      label_name = 'Cup'
      color = color[1]
    else:
      print('Invalid label encountered.')
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=label_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=text_color,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, boxes, labels):
    img = image.copy()
    for box, label in zip(boxes, labels):
      img = visualize_bbox(img, np.array(box), label)
    return img