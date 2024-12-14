import numpy as np
import torch

from ..utils.data_loader import create_data_loader


def accuracy(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

def train_loop(model,config,train_loader,optimizer,criterion):
  model.train()
  
  train_accuracy = 0.0
  train_losses = []
  

  for inputs, labels in train_loader:
    inputs, labels = inputs.to(config['gpu']['device']), labels.to(config['gpu']['device'])

    outputs, h = model(inputs)
    loss = criterion(outputs, labels) 

    optimizer.zero_grad()
    loss.backward() 
    optimizer.step() 

    train_losses.append(loss.item())
    train_accuracy += accuracy(outputs, labels)

  epoch_train_loss = np.mean(train_losses)
  epoch_train_acc = (train_accuracy/len(train_loader.dataset))*100.0
  return (epoch_train_loss, epoch_train_acc)
