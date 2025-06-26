
from scripts.train import train_loop
from scripts.evaluate import evaluate
from scripts.report import report

from utils.criterion import create_criterion
from utils.optimizer import create_optimizer
from utils.data_loader import create_data_loader

from models.bilstm_self_attention import LSTMModel

from omegaconf import OmegaConf

def main():
    try:

        config = OmegaConf.load('config/config.yaml')
        
        lstm_model = LSTMModel(config)
        lstm_model.to(config['gpu']['device'])

        optimizer = create_optimizer(lstm_model, config)
        criterion = create_criterion(config)
        train_loader,_ = create_data_loader(config)


        for epoch in range(config['training']['epochs']):
            epoch_train_loss, epoch_train_acc = train_loop(lstm_model, config, train_loader, optimizer=optimizer,criterion=criterion)
            print("Epoch",epoch+1,"Training Loss is",epoch_train_loss,"and accuracy is",epoch_train_acc)


        
        loss,accuracy = evaluate(lstm_model, config)
        print("Test Loss and Accuracy are",loss,accuracy)

        report(lstm_model, config)

    except FileNotFoundError:
        print("Error: config/config.yaml not found")
    except Exception as e:
        print(f"Error: {e}")



if __name__ == '__main__':
    main()