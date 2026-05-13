import torch.nn as nn


class TicTacToeNet(nn.Module):
    def __init__(self, hidden_sizes: list[int], input_size: int = 18):
        super(TicTacToeNet, self).__init__()
        
        layers = []
        if hidden_sizes[0] is None:
            self.hidden_sizes = [hidden_sizes, 9]
            layers.append(nn.Linear(input_size, 9))
        else:
            self.hidden_sizes = hidden_sizes
            hidden_sizes = [input_size] + hidden_sizes + [9]
            self.hidden_sizes = hidden_sizes

            for i, o in zip(hidden_sizes[:-1], hidden_sizes[1:]):
                layers.append(nn.Linear(i, o))
                layers.append(nn.ReLU()) # Add activation between layers
            # 3. Remove the last ReLU (usually not needed on the output layer)
            layers.pop()
        
        self.model = nn.Sequential(*layers)

        # self.fc_1 = nn.Linear(9, hidden_sizes[1])
        # self.fc_2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        # self.fc_3 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        # self.fc_4 = nn.Linear(hidden_sizes[3], 9)
        # self.relu = nn.ReLU()


    def forward(self, x):
        y = self.model(x)

        # y = self.relu(self.fc_1(x))
        # y = y + self.relu(self.fc_2(y))
        # y = y + self.relu(self.fc_3(y))
        # y = self.fc_4(y)

        illegal = (x[:, :] != 0)
        y.masked_fill(illegal, float('-inf'))
        return y


class routerNet(nn.Module):
    def __init__(self, hidden_sizes: list[int], input_size: int = 9):
        super(routerNet, self).__init__()
        
        output = 2

        layers = []
        if hidden_sizes[0] is None:
            self.hidden_sizes = [hidden_sizes, output]
            layers.append(nn.Linear(input_size, output))
        else:
            self.hidden_sizes = hidden_sizes
            hidden_sizes = [input_size] + hidden_sizes + [output]
            self.hidden_sizes = hidden_sizes

            for i, o in zip(hidden_sizes[:-1], hidden_sizes[1:]):
                layers.append(nn.Linear(i, o))
                layers.append(nn.ReLU()) # Add activation between layers
            # 3. Remove the last ReLU (usually not needed on the output layer)
            layers.pop()
            
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.model(x)
        return logits
    


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