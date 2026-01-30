# thanks Gemini

import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from models.nn import TicTacToeNet
from data.read_data import get_data

DEVICE = DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = f'data/datasets/tictactoe_best_moves.csv'
X_data, y_data = get_data(path=DATA_PATH)
X_data, y_data = X_data.to(DEVICE), y_data.to(DEVICE)

model = TicTacToeNet(hidden_size=64).to(DEVICE)
summary(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

print("Starting training to achieve 0% error...")
epoch = 0
accuracy = 0

experiment_start_time = datetime.now()
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
    
    if epoch % 300 == 0 or accuracy == 100.0:
        print(f'Epoch [{epoch}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}%')

        if accuracy == 100.0:
            checkpoint_time = datetime.now()
            checkpoint_time_str = checkpoint_time.strftime("%Y-%m-%d-%H:%M:%S")
            checkpoint_path = f'models/checkpoints/nn_model_{checkpoint_time_str}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model saved at: {checkpoint_path}')

print(f"\nSuccess! 0% Error achieved in {epoch} epochs.\n")

summary(model)

