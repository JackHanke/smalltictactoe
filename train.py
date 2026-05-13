from time import time
from datetime import datetime
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.reps import *
from data.game import generate_states_from_root_board
from data.dataset import tttDataset
from models.nn import TicTacToeNet

def train_to_perfection(
        model,
        dataset,
        device,
        max_epochs: int = None,
        save_checkpoint: bool = True,
        name: str = '',
        learning_rate: float = 1e-2,
        weight_decay: float = 0.0,
        patience: int = 1_000,
        one_right_answer: bool = True,
    ):
    model.zero_grad()

    perfection_reached = False

    checkpoint_time = datetime.now()
    checkpoint_time_str = checkpoint_time.strftime("%Y-%m-%d-%H:%M:%S")

    # train the given model on the dataset until perfect accuracy is achieved
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Params: {num_params}')

    criterion = nn.CrossEntropyLoss()
    LEARNING_RATE = learning_rate
    WEIGHT_DECAY = weight_decay
    PATIENCE = patience

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
            X_data = X_data.to(device)
            y_data = y_data.to(device)
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

        if one_right_answer:
            correct += (predicted == y_data).sum().item()
        else:
            correct, total = 0, 0
            for board_str, moves in dataset.all_states.items():
                total += 1

                board_rep = dataset.board_rep_func(board_str=board_str)
                board_tensor = torch.tensor(board_rep).float().unsqueeze(0)
                prediction = torch.argmax(model(board_tensor))

                if prediction in moves:
                    correct += 1
                    
        accuracy = 100 * correct / dataset.num_datapoints

        if epoch % 100 == 0 or accuracy == 100.0:
            print(f'Epoch [{epoch}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}%, {correct} correct, {dataset.num_datapoints - correct}/{dataset.num_datapoints} remaining.')

            if accuracy == 100.0:
                perfection_reached = True
                if save_checkpoint:
                    checkpoint_path = f'models/checkpoints/nn{name}_{model.hidden_sizes}_{num_params}_{checkpoint_time_str}.pth'
                    model.zero_grad() # zero grads for file size
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f'Model saved at: {checkpoint_path}')

        if epoch == max_epochs: return perfection_reached, epoch, accuracy
    return perfection_reached, epoch, accuracy


def wiggle_to_perfection(
        model,
        dataset,
        max_epochs: int = None,
        save_checkpoint: bool = True,
        name: str = '',
        learning_rate: float = 1e-2,
        weight_decay: float = 0.0,
        patience: int = 1_000,
    ):
    # within each training iteration, constantly re evaluate which dataset to use
    # for states with multiple best answers, only punish the logits that are not among the best
    # this requires a custom loss function I think
    # TODO
    pass

def param_acc_curve(
        param_min: int = 2,
        param_max: int = 15,
        epochs: int = 500,
        seed: int = 0,
    ):
    rep_length = 9
    board_rep_func=trinary_board_rep
    torch.manual_seed(seed)

    all_states = generate_states_from_root_board([' '] * 9, 'X')
    fixed_states = {key:value for key, value in all_states.items() if len(value) == 1}
    # nonfixed_states = {key:value for key, value in all_states.items() if len(value) > 1}

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0

    dataset = tttDataset(
        states_dict=fixed_states,
        board_rep_func=board_rep_func,
        len_rep=rep_length,
    )

    all_dataset = []
    for board_str, moves in all_states.items():
        binary_board = board_rep_func(board_str=board_str)
        all_dataset.append(binary_board)
    all_dataset = np.array(all_dataset, dtype=np.int32)
    all_dataset = torch.from_numpy(all_dataset).float()

    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    criterion = nn.CrossEntropyLoss()

    hidden_dims = list(range(param_min, param_max))
    X, Y, Z = [], [], []
    prog_bar = tqdm(hidden_dims)
    for hidden_dim in prog_bar:
        start = time()
        model = TicTacToeNet(hidden_sizes=[hidden_dim], input_size=rep_length)
        X.append(sum(p.numel() for p in model.parameters()))

        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        board_rep_func = trinary_board_rep

        for epoch in range(epochs):
            for (X_data, y_data) in dataloader:
                model.zero_grad()

                outputs = model(X_data)
                #
                loss = criterion(outputs, y_data)
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            predicted = torch.argmax(outputs, dim=1)

            correct = (predicted == y_data).sum().item()

        Y.append(correct/len(dataset))

        correct, total = 0, 0

        predictions = torch.argmax(model(all_dataset), dim=1)

        for idx, (board_str, moves) in enumerate(all_states.items()):
            total += 1
            if predictions[idx] in moves:
                correct += 1

        Z.append(correct/total)

        prog_bar.set_description(f'Seed: {seed}, Hidden: {hidden_dim}')

    return X, Y, Z, model

