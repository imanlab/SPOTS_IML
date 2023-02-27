# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import torch
import torch.nn as nn

class ACTP(nn.Module):
    def __init__(self, device, input_size, tactile_size):
        super(ACTP, self).__init__()
        self.device = device
        self.lstm1 = nn.LSTM(input_size, 400).to(device)  # tactile
        self.lstm2 = nn.LSTM(400+12, 400).to(device)  # tactile
        self.fc1 = nn.Linear(400+tactile_size, 400).to(device)  # tactile + pos
        self.fc2 = nn.Linear(400, tactile_size).to(device)  # tactile + pos
        self.tan_activation = nn.Tanh().to(device)
        self.relu_activation = nn.ReLU().to(device)

    def init_hidden(self, batch_size):
        self.hidden1 = (torch.zeros(1, batch_size, 400, device=torch.device(self.device)), torch.zeros(1, batch_size, 400, device=torch.device(self.device)))
        self.hidden2 = (torch.zeros(1, batch_size, 400, device=torch.device(self.device)), torch.zeros(1, batch_size, 400, device=torch.device(self.device)))

    def forward(self, tactile, state_action, skip_tactile):
        out1, self.hidden1 = self.lstm1(tactile.unsqueeze(0), self.hidden1)
        action_and_tactile = torch.cat((out1.squeeze(), state_action), 1)
        out2, self.hidden2 = self.lstm2(action_and_tactile.unsqueeze(0), self.hidden2)
        lstm_and_prev_tactile = torch.cat((out2.squeeze(), skip_tactile), 1)
        out3 = self.tan_activation(self.fc1(lstm_and_prev_tactile))
        out4 = self.tan_activation(self.fc2(out3))

        return out4