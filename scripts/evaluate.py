
import torch 
import torch.nn as nn
import numpy as np
from ..utils.data_loader import create_data_loader

def create_criterion(config):
    loss_type = config['training'].get('loss_function', 'cross_entropy')
    
    if loss_type == 'CE':
        return nn.CrossEntropyLoss()
   
    elif loss_type == 'BCE':
        return nn.BCELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")
    


def accuracy(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()


def evaluate(model, config):
  model.eval()
  test_accuracy = 0.0
  test_losses = []
  _,test_loader = create_data_loader()
  criterion = create_criterion(config)


  with torch.no_grad():
    for inputs, labels in test_loader:
      inputs, labels = inputs.to(config['gpu']['device']), labels.to(config['gpu']['device'])

      outputs, _ = model(inputs)
      loss = criterion(outputs, labels)

      test_losses.append(loss.item())
      test_accuracy += accuracy(outputs, labels)


  return (np.mean(test_losses), (test_accuracy/len(test_loader.dataset))*100.0)