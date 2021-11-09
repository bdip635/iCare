import torch
from torch.nn import *
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from torchvision.models import wide_resnet50_2, mobilenet_v2
from torchsummary import summary


class Wide_ResNet_with_GRADCAM(Module):
  def __init__(self, loaded_model):
    super(Wide_ResNet_with_GRADCAM, self).__init__()
    
    
    layers= list(loaded_model.children())
    self.CNN= Sequential(*layers[:-2])
    self.pool= layers[-2]
    self.classifier= layers[-1]
    self.gradients= None
  
  def activations_hook(self, grad):
    self.gradients= grad
  
  def get_activations_gradient(self):
    return self.gradients

  def get_activations(self, x):
    return self.CNN(x)
  
  def forward(self, x):
    x= self.CNN(x)
    h= x.register_hook(self.activations_hook)
    x= self.pool(x).view((x.shape[0],-1))
    return self.classifier(x)
  

class FocalLoss(Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.weight= weight

    def forward(self, inputs, targets, alpha=0.8, gamma=5, smooth=1):
        
        targets= torch.unsqueeze(targets, dim=1)
        targets= torch.cat((targets, 1-targets), dim=1).to(torch.float32)
        if self.weight is None:
            BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        else:
            BCE = F.binary_cross_entropy(inputs, targets, weight= self.weight, reduction='mean')
            
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
  
def check_accuracy(scores, targets):

  num_correct=0
  num_samples=0
  _, predictions= scores.max(1)
  num_correct+= (predictions== targets).sum()
  num_samples= predictions.size(0)

  return num_correct/num_samples
 
def save_model_checkpoint(model,file_name):

  checkpoint= {'state_dict': model.state_dict()}
  torch.save(checkpoint,file_name)

def load_model_checkpoint(model, file_name, device):
  check_pt= torch.load(file_name, map_location= torch.device(device))
  model.load_state_dict(check_pt['state_dict'])
  return model

def save_checkpoint(model, optimizer, file_name):

  checkpoint= {'state_dict': model.state_dict(),
             'optimizer_dict': optimizer.state_dict()}
  torch.save(checkpoint,file_name)

def load_checkpoint(model, optimizer, file_name):
  check_pt= torch.load(file_name, map_location= torch.device(device))
  model.load_state_dict(check_pt['state_dict'])
  optimizer.load_state_dict(check_pt['optimizer_dict'])

  return model, optimizer


def get_all_preds(model, loader):
    all_preds = torch.tensor([], device=device)
    all_labels= torch.tensor([], device=device)
    model.eval()
    for images,labels in loader:
        images= images.to(device)
        labels= labels.to(device)
        scores = model(images)
        _, preds= scores.max(1)
        all_preds = torch.cat((all_preds, preds),dim=0)
        all_labels = torch.cat((all_labels, labels),dim=0)
    return all_preds, all_labels

def gradcam(img, model,device, class_label=1):

  model= model.to(device=device)
  img= (img-img.min())/(img.max()-img.min())
  input= torch.tensor([img], dtype=torch.float32).to(device=device)
  input= input.permute(0,3,1,2)
  preds = model(input)
  preds[:,class_label].backward()
  gradients= model.get_activations_gradient()
  pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
  activations = model.get_activations(input).detach()
  for i in range(activations.shape[1]):
    activations[:, i, :, :] *= pooled_gradients[i]
  heatmap = torch.mean(activations, dim=1).squeeze()
  heatmap /= torch.max(heatmap)
  return cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)