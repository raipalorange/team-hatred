import torch
from torch.utils.data import DataLoader, TensorDataset
from .preprocessing import convert_sequences_to_tensor,clean_davidson_tweets


def create_data_loader(config):

    df_train,df_test = clean_davidson_tweets()

    train_data_X = convert_sequences_to_tensor(df_train,config)
    train_data_y = torch.FloatTensor([int(d) for d in df_train['class'].to_numpy()])

    test_data_X = convert_sequences_to_tensor(df_test, config)
    test_data_y = torch.FloatTensor([int(d) for d in df_test['class'].to_numpy()])

    train_data = TensorDataset(train_data_X, train_data_y)
    test_data = TensorDataset(test_data_X, test_data_y)

    return DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True),DataLoader(test_data, batch_size=config['training']['batch_size'], shuffle=False)