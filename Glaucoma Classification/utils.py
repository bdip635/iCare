import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import os
import numpy as np
import math
import cv2

def save_checkpoint(model, optimizer, file_name):
  checkpoint= {'state_dict': model.state_dict(),
             'optimizer_dict': optimizer.state_dict()}
  torch.save(checkpoint, file_name)

def load_checkpoint(model, optimizer, file_name, device):
  check_pt= torch.load(file_name, map_location= torch.device(device))
  model.load_state_dict(check_pt['state_dict'])
  optimizer.load_state_dict(check_pt['optimizer_dict'])
  return model, optimizer

def merge_samples_to_batches(targets, batch_size):
  new_targets = []
  for i in range(batch_size):
    new_targets.append({})
    for k in targets:
      new_targets[i][k] = targets[k][i]
  return new_targets

def process_data(images, targets, batch_size, device):
  images = images/255.0
  targets = merge_samples_to_batches(targets, batch_size)
  images_list = list(image.to(device) for image in images)
  targets_dict = [{k: v.to(device) for k, v in t.items()} for t in targets]
  return images_list, targets_dict