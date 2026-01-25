
import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt

from nn import TicTacToeNet

PATH = f'tictactoe_model.pth'
model = TicTacToeNet(hidden_size=48)
model.load_state_dict(torch.load(PATH))
summary(model)

print(model.model[0].weight.shape)
print(model.model[0].bias.shape)
print(model.model[2].weight.shape)
print(model.model[2].bias.shape)

# layer2 = torch.cat([model.model[2].weight.detach(), model.model[2].bias.detach().unsqueeze(1)], dim=1)
# layer0 = torch.cat([model.model[0].weight.detach(),model.model[0].bias.detach().unsqueeze(1)], dim=1).transpose(0,1)
# arr = torch.cat([layer0, layer2])

arr = torch.cat([model.model[0].weight.detach().transpose(0,1), model.model[2].weight.detach()])

# print(layer0.shape)
# print(layer2.shape)

plt.imshow(arr)
plt.colorbar()
plt.title(f'Weights of network')
plt.show()
