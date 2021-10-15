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
from config import *

# check_pt_file= 'wide_resnet50_2.pth.tar'
# DEVICE= 'cuda' if torch.cuda.is_available else 'cpu'
# IMG_SIZE= (512, 512)
def prediction(input_path, save_path, blend_path):
  model= wide_resnet50_2(num_classes=2).to(device=DEVICE)
  model.fc= Sequential(Linear(model.fc.in_features, 2), Softmax(dim=1)).to(device=DEVICE)
  model= Wide_ResNet_with_GRADCAM(load_model_checkpoint(model, WEIGHTS_FILE, DEVICE))
  image= cv2.resize(cv2.imread(input_path)[:,:,::-1], IMG_SIZE)/255.
  heatmap= gradcam(image, model, device=DEVICE)
  heatmap= cv2.resize(heatmap, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
  blended= cv2.addWeighted(np.uint8(255*image), 0.6, heatmap, 0.4, 0)
  flag1= cv2.imwrite(save_path, heatmap)
  flag2= cv2.imwrite(blend_path, blended)
  if (flag1 and flag2) is True:
    print("Successful")

def main():
    image_path= input("Enter image path: ")
    save_path= input("Enter path for saving heatmap: ")
    blend_path= input("Enter path for saving blended image: ")
    prediction(image_path, save_path, blend_path)

if __name__=='__main__':
    main()
    