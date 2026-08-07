import torch
import torch.nn as nn


class TicTacToeNet(nn.Module):
    def __init__(self,
            hidden_sizes: list[int],
            input_size: int = 18,
            do_illegal_move_masking: bool = True
        ):
        super(TicTacToeNet, self).__init__()
        self.do_illegal_move_masking = do_illegal_move_masking
        self.input_size = input_size
        
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
        # self.layers = layers
        
        self.model = nn.Sequential(*layers)
        # self.relu = nn.ReLU(inplace=True)


        # depth = 13
        # print(f'NOTE YOU HARDCODED THE RESNET ARCH!')
        # self.lay0 = nn.Linear(9,depth)
        # self.lay1 = nn.Linear(depth,depth)
        # self.lay2 = nn.Linear(depth,depth)
        # self.lay3 = nn.Linear(depth,depth)
        # self.lay4 = nn.Linear(depth,depth)
        # self.lay5 = nn.Linear(depth,depth)
        # self.lay6 = nn.Linear(depth,depth)
        # self.lay7 = nn.Linear(depth,depth)
        # self.lay8 = nn.Linear(depth,depth)
        # self.lay9 = nn.Linear(depth,9)


    def forward(self, x):
        y = self.model(x)

        
        # x = torch.nn.functional.relu(self.lay0(x))
        # x = x + torch.nn.functional.relu(self.lay1(x))
        # x = x + torch.nn.functional.relu(self.lay2(x))
        # x = x + torch.nn.functional.relu(self.lay3(x))
        # x = x + torch.nn.functional.relu(self.lay4(x))
        # x = x + torch.nn.functional.relu(self.lay5(x))
        # x = x + torch.nn.functional.relu(self.lay6(x))
        # x = torch.nn.functional.relu(self.lay7(x))
        # x = torch.nn.functional.relu(self.lay8(x))
        # x = torch.nn.functional.relu(self.lay9(x))
        # y = x

        if self.do_illegal_move_masking:
            if self.input_size == 9 or self.input_size == 17:
                illegal = (x[:, :9] != 0)
                y = y.masked_fill(illegal, float('-inf'))
            elif self.input_size == 18:
                illegal = torch.bitwise_or((x[:, :9] != 0), (x[:, 9:] != 0))
                y = y.masked_fill(illegal, float('-inf'))
            else:
                raise Exception('Something went wrong with illegal move masking.')
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
    


# class simpleNet(nn.Module):
#     def __init__(self, size: int):
#         super(simpleNet, self).__init__()
#         self.size = size
#         self.model = nn.Sequential(
#             nn.Linear(self.size, self.size),
#             nn.SiLU(),
#             nn.Linear(self.size, 2),
#         )
#     def forward(self, x):
#         x = self.model(x)
#         return x