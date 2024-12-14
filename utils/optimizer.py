
import torch.optim as optim

def create_optimizer(model, config):

    optimizer_name = config['training']['optimizer']
    learning_rate = config['training']['learning_rate']

    if optimizer_name == 'adam':
        return optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=config['training'].get('weight_decay', 0)
        )
    elif optimizer_name == 'sgd':
        return optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=config['training'].get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    