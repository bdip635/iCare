import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import cv2
import numpy as np
import math
from utils_av import *

image_shape = (512, 512, 3)
num_classes = 3    # Artery, Vein, Background
class_labels = {0:(0,0,0), 1:(255,0,0), 2:(0,0,255)}    # Black - Background, Red - Artery, Blue - Vein
device = 'cuda' if torch.cuda.is_available() else 'cpu'

weights_file = 'weights_unet.pth.tar'
model = UNET(out_channels = 3).to(device=device)
unet = load_checkpoint(model, weights_file, device)   # Loading weights into the model

def prediction_fun(input_path, output_path):
  x = cv2.cvtColor(cv2.resize(cv2.imread(image_path), image_shape[:2]), cv2.COLOR_BGR2RGB)
  x = torch.tensor(x, dtype=torch.float32, device=device)/255.
  x = torch.unsqueeze(x.permute(2,0,1), 0)
  x = x.to(device) #(1, 3, 512, 512)

  unet.eval()
  y = unet(x)
  pred = process_onehot_to_mask(y, class_labels)[0] #(512, 512, 3)
  cv2.imwrite(output_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

if __name__=='__main__':
  image_path = input('Enter input image path:')
  pred_path = input('Enter output image path:')
  prediction_fun(image_path, pred_path)