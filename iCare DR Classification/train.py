import torch, os, cv2, math, pkbar, torchvision
from torch.nn import *
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import wide_resnet50_2
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import  albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import ConfusionMatrixDisplay as CMD
from sklearn.metrics import confusion_matrix as CMT
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, jaccard_score, classification_report
from config import *
from dataset import *
from utils import *
import pkbar

train_loader, val_loader, test_loader= get_dataloaders()
train_dataset, val_dataset= train_loader.dataset, val_loader.dataset
model= wide_resnet50_2(progress=True).to(device=DEVICE)
model.fc= Sequential(Linear(model.fc.in_features, NUM_CLASSES),
                     Softmax(dim=1)).to(device=DEVICE)
training_LOSS= CrossEntropyLoss(weight= torch.Tensor(list(train_dataset.get_class_weights().values())).to(device=DEVICE))
validation_LOSS= CrossEntropyLoss(weight=torch.Tensor(list(val_dataset.get_class_weights().values())).to(device=DEVICE))
optimizer= Adam(model.parameters(), lr=LEARNING_RATE)
scheduler= StepLR(optimizer, step_size=5, gamma=0.8)
    
def train(load=False, weights_file=WEIGHTS_FILE):
    
    train_per_epoch = len(train_loader)
    val_per_epoch = len(val_loader)
    min_loss = math.inf
    
    if load is True:
        model= load_model_checkpoint(model, weights_file, DEVICE)

    print("Starting to train.....\n")
    for epoch in range(NUM_EPOCHS):
      train_losses = []
      train_accs = []
      kbar_train = pkbar.Kbar(target = train_per_epoch, epoch = epoch, num_epochs = NUM_EPOCHS)
      model.train()
      for batch_idx, (data, targets) in enumerate(train_loader):
    
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
    
        scores = model(data)
        train_loss = training_LOSS(scores, targets)
        train_losses.append(train_loss.item())
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    
        train_acc = check_accuracy(scores, targets)
        train_accs.append(train_acc.item())
        kbar_train.update(batch_idx, values=[("loss", train_loss.item()), ("accuracy", train_acc.item())])
      
      mean_train_loss = np.mean(train_losses)
      mean_train_acc = np.mean(train_accs)
      kbar_train.update(train_per_epoch, values=[("loss", mean_train_loss), ("accuracy", mean_train_acc)])#For each epoch
    
      val_losses = []
      val_accs = []
      kbar_val = pkbar.Kbar(target = val_per_epoch, epoch = epoch, num_epochs = NUM_EPOCHS)
      with torch.no_grad():
        model.eval()
        for batch_idx, (data, targets) in enumerate(val_loader):
          data = data.to(device=DEVICE)
          targets = targets.to(device=DEVICE)
    
          scores = model(data)
          val_loss = validation_LOSS(scores, targets)
          val_losses.append(val_loss.item())
    
          val_acc = check_accuracy(scores, targets)
          val_accs.append(val_acc.item())
          kbar_val.update(batch_idx, values=[("val_loss", val_loss.item()), ("val_accuracy", val_acc.item())])
        
        mean_val_loss = np.mean(val_losses)
        mean_val_acc = np.mean(val_accs)
        kbar_val.update(val_per_epoch, values=[("val_loss", mean_val_loss), ("val_accuracy", mean_val_acc)])#For each epoch
        
        if mean_val_loss < min_loss:
          min_loss = mean_val_loss
          print('\nImproved validation Loss: {}'.format(min_loss))
          save_checkpoint(model, optimizer, weights_file)
          print('Model saved to {}\n'.format(weights_file))
    
    print("....Printing Metrics....")
    plot_metrics()
    
def plot_metrics():
    net= load_model_checkpoint(model, WEIGHTS_FILE, DEVICE)
    with torch.no_grad():
        all_preds, all_labels = get_all_preds(net.to(device=DEVICE), test_loader)

    cmt= CMT(all_labels.cpu(), all_preds.cpu())
    prec, rec, f1,_= precision_recall_fscore_support(all_labels.cpu(), all_preds.cpu())
    j_score= jaccard_score(all_labels.cpu(), all_preds.cpu(), average=None)
    acc= accuracy_score(all_labels.cpu(), all_preds.cpu())
    mean_f1= f1_score(all_labels.cpu(), all_preds.cpu(), average='weighted')
    avg_j_score= jaccard_score(all_labels.cpu(), all_preds.cpu(), average='weighted')
    avg_prec, avg_rec, _, _=precision_recall_fscore_support(all_labels.cpu(), all_preds.cpu(), average='weighted')
    
    names= ['DR=0', 'DR=1']
    disp= CMD(cmt/np.sum(cmt), names)
    disp.plot(cmap='Blues')
    print('Classification Report\n {}'.format(classification_report(all_labels.cpu(), all_preds.cpu())))
    print(f"Average Precision: {avg_prec}")
    print(f"Average Recall: {avg_rec}")
    print(f"Average F1 score: {mean_f1}")
    print(f"Average Jaccard Score: {avg_j_score}")

if __name__=='__main__':
    ch= input('Do you want to load the default pretrained model for further training?: ')
    if ch=='y':
        train(load=True)
    else:
        path= input("Enter the path o the weights file: ")
        train(load=False, weights_file=path)