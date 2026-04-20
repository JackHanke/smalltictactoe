import torch.nn as nn

class TicTacToeNet(nn.Module):
    def __init__(self, hidden_size: int):
        super(TicTacToeNet, self).__init__()
        self.hidden_size = hidden_size
        self.model = nn.Sequential(
            nn.Linear(9, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 9),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
class simpleNet(nn.Module):
    def __init__(self, size: int):
        super(simpleNet, self).__init__()
        self.size = size
        self.model = nn.Sequential(
            nn.Linear(self.size, self.size),
            nn.SiLU(),
            nn.Linear(self.size, 2),
        )
    def forward(self, x):
        x = self.model(x)
        return x