import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
import numpy as np
import math
import cv2

class Conv_Block(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, activation='relu', mode='normal'):
    super(Conv_Block, self).__init__()
    self.mode_dict = nn.ModuleDict({
        'normal': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
        'separable': nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels),
                                   nn.Conv2d(in_channels, out_channels, kernel_size=1))
    })
    self.activations_dict = nn.ModuleDict({
        'relu': nn.ReLU(inplace=True),
        'softmax': nn.Softmax(),
        'softmax2d': nn.Softmax2d(),
        'sigmoid': nn.Sigmoid()
    })
    self.conv = self.mode_dict[mode]
    self.bn = nn.BatchNorm2d(out_channels)
    self.act = self.activations_dict[activation]

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.act(x)
    return x

class REDNet(nn.Module):
  def __init__(self, in_features, connection_type = 'orig'):
    super(REDNet, self).__init__()
    self.conv_layer = Conv_Block(in_features, in_features, kernel_size=3, stride=1, padding=1)
    self.conv_transpose_layer = nn.Sequential(nn.ConvTranspose2d(in_features, in_features, kernel_size=3, stride=1, padding=1), #(s,p) = (3,1) or (1,0)
                                              nn.BatchNorm2d(in_features),
                                              nn.ReLU(inplace=True))
    connectors={}
    connectors['orig'] = [-2,-2,-2,-2]
    connectors['t1'] = [-1,-1,1,0]
    connectors['t2'] = [0,1,2,3]
    connectors['t3'] = [-1,0,-1,2]
    self.connectors = connectors[connection_type]
    
  def forward(self, x):
    layers_list = [x]
    for i in self.connectors[:2]:
      if i<=-1:
        t = self.conv_layer(layers_list[-1])
      else:
        t = self.conv_layer(layers_list[-1]) + layers_list[i]
      layers_list.append(t)
    for i in self.connectors[2:]:
      if i==-1:
        t = self.conv_transpose_layer(layers_list[-1])
      elif i==-2:
        t = self.conv_layer(layers_list[-1])
      else:
        t = self.conv_transpose_layer(layers_list[-1]) + layers_list[i]
      layers_list.append(t)
    return layers_list[-1]

class RED_RCNN(nn.Module):
  def __init__(self, num_classes, connection_type='t3'):
    super(RED_RCNN, self).__init__()
    self.pmodel = maskrcnn_resnet50_fpn(pretrained=True, progress=True, pretrained_backbone=True, trainable_backbone_layers=None)

    self.in_features_box = self.pmodel.roi_heads.box_predictor.cls_score.in_features
    self.in_features_mask = self.pmodel.roi_heads.mask_predictor.mask_fcn_logits.in_channels
    self.hidden_layers = 256

    self.pmodel.roi_heads.box_predictor = FastRCNNPredictor(self.in_features_box, num_classes)
    self.pmodel.roi_heads.mask_head = REDNet(self.in_features_mask, connection_type)
    self.pmodel.roi_heads.mask_predictor = MaskRCNNPredictor(self.in_features_mask, self.hidden_layers, num_classes)
    
  def forward(self, x, y = None, mode = 'eval'):
    if mode == 'train':
      self.train()
      y = self.pmodel(x, y)
    elif mode == 'eval':
      self.eval()
      y = self.pmodel(x)
    else:
      print('Invalid mode')
    return y

def load_checkpoint(model, file_name, device):
  check_pt = torch.load(file_name, map_location= torch.device(device))
  model.load_state_dict(check_pt['state_dict'])
  return model

def process_data(images, device):
  images_list = list(image.to(device) for image in images)
  return images_list

def post_process(x, thresh):
  y = np.zeros_like(x)
  y[x>thresh] = 1.0
  return y

def find_vcdr(boxes, labels):
  dic = {int(labels[i]): boxes[i].cpu().detach().numpy() for i in range(len(labels))}
  od = dic[1][3] - dic[1][1]
  oc = dic[2][3] - dic[2][1]
  return oc/od