import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 500)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(500, 500)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(500, 500)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(500, 1)
        self.dropout4 = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        x = self.dropout4(x)
        x = self.fc4(x)
        
        return x