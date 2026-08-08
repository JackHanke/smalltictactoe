from time import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.reps import *
from data.game import generate_states_from_root_board
from data.dataset import tttDataset
from models.nn import TicTacToeNet

# def one_among_many_loss(logits, moves_mask):
#     # NOTE removes row of all True, doesnt matter what the network does
#     # TODO there is a better fix for this, pick just corners or something
#     logits = logits.clone()[1:]
#     moves_mask = moves_mask.clone()[1:]

#     mask_val = -float('inf')
#     good_logits = logits.masked_fill(~moves_mask, mask_val)
#     bad_logits = logits.masked_fill(moves_mask, mask_val)

#     # loss = (torch.max(bad_logits, dim=1).values - torch.max(good_logits, dim=1).values).mean()
#     loss = (torch.mean(bad_logits, dim=1) - torch.mean(good_logits, dim=1)).mean()
#     return loss

def one_among_many_loss(logits, moves_mask, legal_moves_mask):
    # identify rows in which any legal move can be played
    row_mask = (~(moves_mask == legal_moves_mask).all(dim=1)).nonzero().squeeze(1)

    logits = logits[row_mask]
    moves_mask = moves_mask[row_mask]
    legal_moves_mask = legal_moves_mask[row_mask]

    # find rows in which only legal moves are the moves_mask, and remove from loss
    # 

    logits = torch.nn.functional.log_softmax(logits, dim=1)

    # mask_val = -float('inf')
    # good_logits = logits.masked_fill(~moves_mask, mask_val)
    # bad_logits = logits.masked_fill(moves_mask, mask_val)

    good_logits = logits * moves_mask
    bad_logits = logits * (1-moves_mask) * legal_moves_mask

    loss = (torch.max(bad_logits, dim=1).values - torch.max(good_logits, dim=1).values).mean()
    # loss = (torch.mean(bad_logits, dim=1) - torch.mean(good_logits, dim=1)).mean()

    term_1 = (bad_logits.sum(dim=1) / (1-moves_mask).sum(dim=1).clamp(min=1))
    term_2 = (good_logits.sum(dim=1) / moves_mask.sum(dim=1).clamp(min=1))
    # print(term_1.mean().item(), term_2.mean().item())
    loss =  term_1 - term_2
    # return loss.mean(), term_1.mean().item(), term_2.mean().item()
    return loss.mean()

def train_to_perfection(
        model,
        dataset,
        device,
        max_epochs: int = None,
        save_checkpoint: bool = True,
        name: str = '',
        learning_rate: float = 1e-2,
        weight_decay: float = 0.0,
        one_right_answer: bool = True,
        verbose: bool = False,
    ):
    with open("data/datasets/jsons/all_states.json", "r") as file:
        all_states_dict = json.load(file)
    moves_mask = []
    for key, moves in all_states_dict.items():
        row = [0 for _ in range(9)]
        for move in moves:
            row[move] = 1
        moves_mask.append(row)
    # moves_mask = torch.tensor(moves_mask, dtype=torch.bool, device=device)
    moves_mask = torch.tensor(moves_mask, device=device)

    boards_lst = [key for key in all_states_dict]

    model.zero_grad()

    perfection_reached = False

    checkpoint_time = datetime.now()
    checkpoint_time_str = checkpoint_time.strftime("%Y-%m-%d-%H:%M:%S")

    # train the given model on the dataset until perfect accuracy is achieved
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    num_params = sum(p.numel() for p in model.parameters())
    # print(f'Params: {num_params}')

    # criterion = nn.CrossEntropyLoss()
    criterion = one_among_many_loss
    LEARNING_RATE = learning_rate
    WEIGHT_DECAY = weight_decay

    # configs_str = f'''Configs:
    # LEARNING_RATE: {LEARNING_RATE}
    # WEIGHT_DECAY: {WEIGHT_DECAY}
    # '''
    # print(configs_str)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.9,
        patience=50,
        min_lr=1e-5,
    )
    # scheduler = optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=1.0,
    #     end_factor=0.001,
    #     total_iters=max_epochs
    # )

    # train to dataset where there is only one option

    epoch, accuracy = 0, 0
    while accuracy < 100.0:
        epoch += 1

        for (X_data, y_data) in dataloader:
            X_data = X_data.to(device)
            y_data = y_data.to(device)
            #
            outputs = model(X_data)
            #
            # loss = criterion(outputs, y_data)
            legal_moves_mask = (X_data == 0)
            loss = criterion(outputs, moves_mask, legal_moves_mask)
            # loss, term_1, term_2 = criterion(outputs, moves_mask, legal_moves_mask)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # scheduler.step()
        scheduler.step(loss)
        
        predicted = torch.argmax(outputs, dim=1)


        correct = 0
        for board_idx, (_, moves) in enumerate(all_states_dict.items()):
            if predicted[board_idx] in moves:
                correct += 1
        current_dataset_correct = correct

        # current_dataset_correct = (predicted == y_data).sum().item()
        # correct = current_dataset_correct
        # if not one_right_answer:
        #     # find points predictions that are wrong for the training dataset but are still correct among all possible moves
        #     wrong_indices = torch.nonzero(predicted != y_data, as_tuple=False)
        #     all_states_correct = 0
        #     # NOTE assumes states are in the same order
        #     for idx in wrong_indices:
        #         moves = all_states_dict[boards_lst[idx]]
        #         if predicted[idx] in moves:
        #             all_states_correct += 1
        #     correct += all_states_correct

        accuracy = 100 * correct / dataset.num_datapoints

        if epoch % 1 == 0 or accuracy == 100.0:
            if verbose:
                print(f'Epoch [{epoch}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}%, Correct (Dataset/All): {current_dataset_correct} / {correct}, {dataset.num_datapoints - correct}/{dataset.num_datapoints} remaining.')
                # print(f'Term 1,2: {term_1, term_2}')

            if accuracy == 100.0:
                perfection_reached = True
                if save_checkpoint:
                    checkpoint_path = f'models/checkpoints/nn{name}_{model.hidden_sizes}_{num_params}_{checkpoint_time_str}.pth'
                    model.zero_grad() # zero grads for file size
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f'Model saved at: {checkpoint_path}')
                    print(f'Loss: {loss.item():.5f}')

        if epoch == max_epochs: return perfection_reached, epoch, accuracy
    
    return perfection_reached, epoch, accuracy

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

