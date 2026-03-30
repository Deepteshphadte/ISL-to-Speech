import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=126, hidden_size=128, num_classes=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])