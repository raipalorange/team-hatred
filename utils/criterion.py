import torch.nn as nn


def create_criterion(config):
    loss_type = config['training'].get('loss_function', 'BCE')
    
    if loss_type == 'CE':
        return nn.CrossEntropyLoss()
   
    elif loss_type == 'BCE':
        return nn.BCELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")
    