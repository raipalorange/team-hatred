
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
  def __init__(self,config):
    '''
    lstm_input_dim: number of values in an embedding
    '''
    super(LSTMModel,self).__init__()

    self.num_stacked_layers = config['model']['num_stacked_layers']
    self.hidden_size = config['model']['hidden_size']
    self.self_attention = nn.MultiheadAttention(embed_dim=config['model']['input_size'], num_heads=config['model']['heads'], batch_first=True,dropout=config['model']['dropout'])

    self.lstm = nn.LSTM(
        input_size = config['model']['input_size'],
        hidden_size = config['model']['hidden_size'],
        num_layers = config['model']['num_stacked_layers'],
        batch_first = True,
        bidirectional=True,
        dropout=config['model']['dropout'],
      )
    
    self.dropout = nn.Dropout(config['model']['drop_prob'])

    self.fc = nn.Linear(config['model']['hidden_size']*2, config['model']['hidden_size']) # changed
    self.fc2 = nn.Linear(config['model']['hidden_size'], 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x, config):

    batch_size = x.size(0)

    x, _ = self.self_attention(x, x, x)


    h0 = torch.zeros((2*self.num_stacked_layers, batch_size, self.hidden_size)).to(config['gpu']['device'])
    c0 = torch.zeros((2*self.num_stacked_layers, batch_size, self.hidden_size)).to(config['gpu']['device'])

 
    lstm_out, hidden = self.lstm(x, (h0, c0))


    lstm_out = self.dropout(lstm_out)
    fc_out = self.fc(lstm_out[:, -1, :])
    fc2_out = self.fc2(fc_out) 

    sigmoid_out = self.sigmoid(fc2_out) 

    return sigmoid_out.squeeze(), hidden