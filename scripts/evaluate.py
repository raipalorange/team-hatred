
import torch 
import torch.nn as nn
import numpy as np
from utils.data_loader import create_data_loader

from utils.criterion import create_criterion
    


def accuracy(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()


def evaluate(model, config):
  model.eval()
  test_accuracy = 0.0
  test_losses = []
  _,test_loader = create_data_loader(config)
  criterion = create_criterion(config)


  with torch.no_grad():
    for inputs, labels in test_loader:
      inputs, labels = inputs.to(config['gpu']['device']), labels.to(config['gpu']['device'])

      outputs, _ = model(inputs,config)
      loss = criterion(outputs, labels)

      test_losses.append(loss.item())
      test_accuracy += accuracy(outputs, labels)


  return (np.mean(test_losses), (test_accuracy/len(test_loader.dataset))*100.0)