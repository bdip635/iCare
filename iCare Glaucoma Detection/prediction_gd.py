import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import cv2
import numpy as np
import math
from utils_gd import *

image_shape = (512, 512, 3)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

weights_file = 'weights_redrcnn.pth.tar'
model = RED_RCNN(num_classes=3, connection_type='t3').to(device=device)
net = load_checkpoint(model, weights_file, device)   # Loading weights into the model

def prediction_fun(input_path, output_path):
  x = cv2.cvtColor(cv2.resize(cv2.imread(image_path), image_shape[:2]), cv2.COLOR_BGR2RGB)
  x = torch.tensor(x, dtype=torch.float32, device=device)/255.
  x = torch.unsqueeze(x.permute(2,0,1), 0)
  x = x.to(device) #(1, 3, 512, 512)

  x = process_data(x, device) # x is a list containing 1 tensor of size (3,512,512).
  net.eval()
  y = net(x, mode='eval')

  cdr = find_vcdr(y[0]['boxes'], y[0]['labels'])
  od = y[0]['masks'][0, 0].cpu().detach().numpy()
  oc = y[0]['masks'][1, 0].cpu().detach().numpy()

  t = 0.9  # Keeping threshold of 0.9 for the grayscale OD, OC masks
  od = post_process(od, t)
  oc = post_process(oc, t)
  
  output = np.concatenate((od, np.ones((512,1)), oc), axis=1)
  cv2.imwrite(output_path, (output*255).astype(np.uint8))
  return cdr

if __name__=='__main__':
  image_path = input('Enter input image path: ')
  pred_path = input('Enter output image path: ')
  cdr = prediction_fun(image_path, pred_path)
  print('\nVertical Cup-to-Disc Ratio:', cdr)