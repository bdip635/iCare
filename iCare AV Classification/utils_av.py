import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import cv2
import numpy as np
import math

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=3, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2,))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = nn.Sequential(nn.Conv2d(features[-1], features[-1]*2, 3, 1, 1, bias=False),
                                        nn.BatchNorm2d(features[-1]*2),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.2),
                                        nn.Conv2d(features[-1]*2, features[-1]*2, 3, 1, 1, bias=False),
                                        nn.BatchNorm2d(features[-1]*2),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.2))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sftmx = nn.Softmax(dim=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        x = self.final_conv(x)
        x = self.sftmx(x)
        return x

def load_checkpoint(model, file_name, device):
  check_pt = torch.load(file_name, map_location= torch.device(device))
  model.load_state_dict(check_pt['state_dict'])
  return model

def process_onehot_to_mask(onehot, class_labels):
  onehot = onehot.permute(0, 2, 3, 1).cpu().detach().numpy()
  output = np.zeros(onehot.shape[:3]+(3,))
  single_layer = np.argmax(onehot, axis=-1)
  
  for k in class_labels.keys():
    output[single_layer==k] = class_labels[k]
  output = np.uint8(output)
  return output