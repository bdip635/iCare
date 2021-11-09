import torch
from torch.nn import *
import torch.nn.functional as F
import torchvision
from torchvision.models import wide_resnet50_2, mobilenet_v2
import  albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
from utils import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE= (512, 512)
check_pt_file= 'weights/wide_resnet50_2.pth.tar'

def prediction(input_path, save_path):
  IMG_SIZE= (512, 512)
  check_pt_file= 'weights/wide_resnet50_2.pth.tar'
  model= wide_resnet50_2(num_classes=2).to(device)
  model.fc= Sequential(Linear(model.fc.in_features, 2), Softmax(dim=1)).to(device=device)
  model= Wide_ResNet_with_GRADCAM(load_model_checkpoint(model, check_pt_file, device))
  image= cv2.resize(cv2.imread(input_path)[:,:,::-1], IMG_SIZE)/255.
  heatmap= gradcam(image, model, device)
  heatmap= cv2.resize(heatmap, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
  blended= cv2.addWeighted(np.uint8(255*image), 0.6, heatmap, 0.4, 0)
  flag= cv2.imwrite(save_path, blended)
  if flag is True:
    print("Successful")