from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from data.dataset import tttDataset

def train(model, path: str, len_rep: int):
    dataset = tttDataset(path=path, len_rep=len_rep)
    dataloader = DataLoader(dataset, batch_size=4520, shuffle=False)

    num_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss()
    LEARNING_RATE = 1e-1
    WEIGHT_DECAY = 0
    PATIENCE=500

    configs_str = f'''Configs:
    LEARNING_RATE: {LEARNING_RATE}
    WEIGHT_DECAY: {WEIGHT_DECAY}
    PATIENCE: {PATIENCE}
    '''
    print(configs_str)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=PATIENCE
    )

    epoch, accuracy = 0, 0
    while accuracy < 100.0:
        epoch += 1

        correct = 0
        for (X_data, y_data) in dataloader:
            #
            outputs = model(X_data)
            #
            loss = criterion(outputs, y_data)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # correct += ((outputs >= 0.5) == (y_data == 1)).all(dim=1).sum().item()
        
        scheduler.step(metrics=loss)
        
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == y_data).sum().item()
        accuracy = 100 * correct / dataset.num_datapoints

        if epoch % 10 == 0 or accuracy == 100.0:
            print(f'Epoch [{epoch}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}%, {dataset.num_datapoints - correct}/{dataset.num_datapoints} remaining.')

            if accuracy == 100.0:
                checkpoint_time = datetime.now()
                checkpoint_time_str = checkpoint_time.strftime("%Y-%m-%d-%H:%M:%S")
                checkpoint_path = f'models/checkpoints/model_{num_params}_{checkpoint_time_str}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                print(f'Model saved at: {checkpoint_path}')
