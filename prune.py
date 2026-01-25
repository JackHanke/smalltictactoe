## thanks Gemini

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torchsummary import summary

from nn import TicTacToeNet
from data import get_data

X_data, y_data = get_data()

PATH = f'tictactoe_model.pth'
model = TicTacToeNet()
model.load_state_dict(torch.load(PATH))

outputs = model(X_data)
_, predicted = torch.max(outputs.data, 1)
correct = (predicted == y_data).sum().item()
accuracy = 100 * correct / len(y_data)

summary(model)
print(f'Model loaded at {PATH} is {accuracy}% accurate')

parameters_to_prune = (
    (model.model[0], 'weight'),
    (model.model[2], 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.05, 
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epoch = 0
accuracy = 0

while accuracy < 100.0:
    epoch += 1
    
    # Forward pass
    outputs = model(X_data)
    loss = criterion(outputs, y_data)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == y_data).sum().item()
    accuracy = 100 * correct / len(y_data)
    
    if epoch % 100 == 0 or accuracy == 100.0:
        print(f'Epoch [{epoch}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

print(f"\nSuccess! 0% Error achieved in {epoch} epochs.\n")

for module, name in parameters_to_prune:
    prune.remove(module, name)

torch.save(model.state_dict(), 'tictactoe_model_pruned.pth')

