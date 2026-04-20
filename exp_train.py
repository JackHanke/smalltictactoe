## NN experiment training loop
from tqdm import tqdm
from time import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from data.dataset import binDataset
from models.nn import simpleNet

def train_idividual_num(
        model,
        num: int,
        length: int,
        lr: float,
    ):
    ''' train a given model for dataset indexed by num written as length length binary string '''

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = binDataset(num=num, length=length)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # num_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss()
    LEARNING_RATE = lr
    WEIGHT_DECAY = 0

    configs_str = f'''Configs:
    LEARNING_RATE: {LEARNING_RATE}
    WEIGHT_DECAY: {WEIGHT_DECAY}
    '''
    # print(configs_str)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    epoch, accuracy = 0, 0
    while accuracy < 100.0:
        epoch += 1

        correct = 0
        for (X_data, y_data) in dataloader:
            optimizer.zero_grad()
            X_data = X_data.to(DEVICE)
            y_data = y_data.to(DEVICE)
            #
            outputs = model(X_data)
            #
            loss = criterion(outputs, y_data)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == y_data).sum().item()
        accuracy = 100 * correct / len(dataset)

        # if epoch % 10 == 0 or accuracy == 100.0:
            # print(f'Epoch [{epoch}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}%, {dataset.num_datapoints - correct}/{dataset.num_datapoints} remaining.')

        if accuracy == 100.0:
            return epoch
                

def experiment(
        length: int,
        lr: float,
        num_seeds: int = 5,
    ):
    ''' how long does it take a basic neural network to learn a synthetic binary dataset, full batch'''

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}
    for seed in range(num_seeds):
        exp_loop_start = time()
        prog_bar = tqdm(range(2**(2**length)))
        for num in prog_bar:
            # init model with specific seed
            torch.manual_seed(seed)
            model = simpleNet(size=length).to(DEVICE)
            # print([thing for thing in model.parameters()][0])

            epochs = train_idividual_num(
                model=model,
                num=num,
                length=length,
                lr=lr,
            )

            try:
                results[num] += epochs / num_seeds
            except KeyError:
                results[num] = epochs / num_seeds

            prog_bar.set_description(f'Seed {seed+1}/{num_seeds} Number: {num}')

    for key, val in results.items():
        key_bin = format(key, f'0{2**length}b')
        print(f'{key} : {key_bin} = {val:.1f}')

    return results


if __name__ == '__main__':
    results = experiment(length=2)
    pass