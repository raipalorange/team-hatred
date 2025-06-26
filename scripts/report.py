from sklearn.metrics import classification_report
import torch
from utils.data_loader import create_data_loader

def report(lstm_model,config):

    _,test_loader = create_data_loader(config)

    with torch.no_grad():
        y_true = []
        y_pred = []
        for inputs, labels in test_loader:
            outputs,_ = lstm_model(inputs,config)
            predicted= torch.round(outputs.squeeze())

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    print(classification_report(y_true, y_pred))