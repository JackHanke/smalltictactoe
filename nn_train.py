## thanks Gemini

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import pandas as pd
import numpy as np

from nn import TicTacToeNet
from data import get_data

X_data, y_data = get_data()

PATH = 'tictactoe_model.pth'
model = TicTacToeNet(hidden_size=48)
summary(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-6)

print("Starting training to achieve 0% error...")
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

summary(model)

torch.save(model.state_dict(), PATH)
print(f'Model saved at: {PATH}')