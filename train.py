from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from data.game import generate_states_from_root_board
from data.dataset import tttDataset

def train_to_perfection(
        model,
        dataset,
        max_epochs: int = None,
        save_checkpoint: bool = True,
        name: str = '',
    ):

    checkpoint_time = datetime.now()
    checkpoint_time_str = checkpoint_time.strftime("%Y-%m-%d-%H:%M:%S")

    # train the given model on the dataset until perfect accuracy is achieved
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Params: {num_params}')

    criterion = nn.CrossEntropyLoss()
    LEARNING_RATE = 1e-2
    WEIGHT_DECAY = 0
    PATIENCE=1_000

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

    # train to dataset where there is only one option

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
        
        predicted = torch.argmax(outputs, dim=1)

        correct += (predicted == y_data).sum().item()
        accuracy = 100 * correct / dataset.num_datapoints

        if epoch % 100 == 0 or accuracy == 100.0:
            print(f'Epoch [{epoch}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}%, {dataset.num_datapoints - correct}/{dataset.num_datapoints} remaining.')

            if accuracy == 100.0 and save_checkpoint:
                
                checkpoint_path = f'models/checkpoints/nn{name}_{model.hidden_sizes}_{num_params}_{checkpoint_time_str}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                print(f'Model saved at: {checkpoint_path}')
