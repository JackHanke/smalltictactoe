import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.reps import *
from data.game import generate_states_from_root_board
from data.dataset import tttDataset
from models.nn import TicTacToeNet
from train import train_to_perfection

def nn_friendly(
        device,
        all_states: dict = None,
        hidden_dim: list = [45],
        max_epochs: int = None,
    ):
    # find dataset seeds that are easy for a neural network to model

    if all_states is None:
        all_states = generate_states_from_root_board([' '] * 9, 'X')

    rep_length = 9
    board_rep_func = trinary_board_rep

    all_dataset = tttDataset(
        states_dict=all_states,
        board_rep_func=board_rep_func,
        len_rep=rep_length,
    )
    # begin by assigning fixed_states to all moves with 1 option
    fixed_states = {key:value for key, value in all_states.items() if len(value) == 1}

    num_wrong = len(all_dataset)
    iteration = 0
    print(f'Iteration {iteration} length fixed states: {len(fixed_states)} / {len(all_states)}')
    while num_wrong != 0:
        iteration += 1

        model = TicTacToeNet(hidden_sizes=hidden_dim, input_size=rep_length).to(device)

        dataset = tttDataset(
            states_dict=fixed_states,
            board_rep_func=board_rep_func,
            len_rep=rep_length
        )
        
        train_to_perfection(
            model=model,
            dataset=dataset,
            save_checkpoint=False,
            one_right_answer=True,
            max_epochs=max_epochs,
            device=device,
        )

        # evaluate how this model does on the full dataset, creating new options for dataset seeds
        new_options = {}
        new_fixed_states = {}

        total_seed_options = 1
        correct, num_wrong, total = 0, 0, 0
        for board_str, moves in all_states.items():
            if board_str not in fixed_states:
                total += 1
                board_rep = board_rep_func(board_str=board_str)

                board_tensor = torch.tensor(board_rep).float().unsqueeze(0).to(device)
                prediction = torch.argmax(model(board_tensor))
                if prediction in moves:
                    correct += 1
                    new_options[board_str] = [prediction.item()]
                    new_fixed_states[board_str] = [prediction.item()]
                else:
                    num_wrong += 1
                    new_options[board_str] = moves
                    total_seed_options *= len(moves)
            else:
                new_options[board_str] = fixed_states[board_str]
                new_fixed_states[board_str] = fixed_states[board_str]

        fixed_states = new_fixed_states
        
        print(f'Iteration {iteration} correct: {correct}, total: {total}, acc: {100*correct/total:.4f}% among remaining')
        print(f'Iteration {iteration} total seeds remaining: {total_seed_options:,}')
        if total_seed_options < 1_000:
            # print('new options for seeds:\n', new_options.values())
            if hidden_dim is None: name='linear'
            else: name=''

            with open(f'data/datasets/jsons/nn_friendly_dataset.json', 'w') as fp:
                json.dump(new_options, fp)
            break
        print(f'Iteration {iteration} length fixed states: {len(fixed_states)} / {len(all_states)}, {len(all_states)-len(fixed_states)} left.')

    # print(f' {[opts for opts in new_options if len(opts) > 1]}')
    # print(':\n', new_options)

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # with open(f"data/residual_2162_dataset.json", "r") as file:
    #     states_dict = json.load(file)
    
    states_dict = None

    nn_friendly(
        hidden_dim=[31], 
        max_epochs=5_000, 
        all_states=states_dict,
        device=DEVICE,
    )