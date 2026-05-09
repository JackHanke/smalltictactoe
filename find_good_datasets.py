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
from torchsummary import summary

from data.reps import *
from data.game import generate_states_from_root_board
from data.dataset import tttDataset
from models.nn import TicTacToeNet
from train import train_to_perfection

def svm_fit_per_output(max_iter: int = 250):
    ''' fit an svm to each output, find the lowest classification '''
    print(f'MAX_ITER: {max_iter}')

    least_wrong = 947 # seed = 14, checked up to 5000, many seeds tie this value
    tied_seeds = []
    prog_bar = tqdm(range(1500, 5000))
    prog_bar.set_description(f'Best_seed: {14} Least wrong: {947}')
    for seed in prog_bar:
        # generate a dataset for a given seed
        states = generate_states_from_root_board(
            board=['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
            player='O',
            seed=seed,
        )
        states[' '*9] = ('X', [0])
        # turn the dataset into matrices
        X, Y = [], []
        for board_str, (player, move) in states.items():
            binary_board = binary_board_rep(board_str=board_str)
            one_hot_moves = one_neg_one_move_rep(move=move[0])
            X.append(binary_board)
            Y.append(one_hot_moves)

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)

        total_wrong = 0
        for idx in range(9):
            clf = LinearSVC(max_iter=max_iter)
            clf.fit(X, Y[:, idx])
            # print(f'coefs: {clf.coef_}')
            Y_pred = clf.predict(X)
            num_wrong = (Y_pred != Y[:, idx]).sum()
            total_wrong += num_wrong

        if total_wrong < least_wrong:
            tied_seeds = []
            least_wrong = total_wrong
            prog_bar.set_description(f'Best_seed: {seed} Least wrong: {total_wrong}')
        elif total_wrong == least_wrong:
            tied_seeds.append(seed)
            # print(f'Seed {seed} tied for best at {total_wrong}!')

    print(f'Tied Seeds: {tied_seeds}')

def dec_tree_fit_per_output():
    ''' fit an decision tree to each dataset, find the tree with smallest node '''
    least_nodes = float('inf') 
    tied_seeds = []
    prog_bar = tqdm(range(0, 1_000)) # seed 693, Nodes: 1019 Leaves: 510 Depth: 17
    # prog_bar.set_description(f'Best_seed: {14} Least wrong: {947}')
    for seed in prog_bar:
        # generate a dataset for a given seed
        states = generate_states_from_root_board(
            board=['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
            player='O',
            seed=seed,
        )
        states[' '*9] = ('X', [0])
        # turn the dataset into matrices
        X, Y = [], []
        for board_str, (player, move) in states.items():
            binary_board = binary_board_rep(board_str=board_str)
            # one_hot_moves = one_neg_one_move_rep(move=move[0])
            X.append(binary_board)
            Y.append(move[0])

        X = np.array(X, dtype=np.int32)
        Y = np.array(Y, dtype=np.int32)

        clf = DecisionTreeClassifier()

        clf.fit(X, Y)
        # Y_pred = clf.predict(X)

        num_nodes = clf.tree_.node_count

        if num_nodes < least_nodes:
            least_nodes = num_nodes
            prog_bar.set_description(f'Best_seed: {seed} Nodes: {num_nodes} Leaves: {clf.get_n_leaves()} Depth: {clf.get_depth()}')
                
    # print(f'Tied Seeds: {tied_seeds}')

def nn_friendly(
        all_states: dict = None,
        hidden_dim: int = 45,
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

        model = TicTacToeNet(hidden_sizes=[hidden_dim], input_size=rep_length)

        dataset = tttDataset(
            states_dict=fixed_states,
            board_rep_func=board_rep_func,
            len_rep=rep_length
        )
        
        train_to_perfection(
            model=model,
            dataset=dataset,
            save_checkpoint=False,
            max_epochs=max_epochs,
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

                board_tensor = torch.tensor(board_rep).float().unsqueeze(0)
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

            with open(f'{name}_{rep_length}_seed_options.json', 'w') as fp:
                json.dump(new_options, fp)
            break
        print(f'Iteration {iteration} length fixed states: {len(fixed_states)} / {len(all_states)}, {len(all_states)-len(fixed_states)} left.')

    # print(f' {[opts for opts in new_options if len(opts) > 1]}')
    # print(':\n', new_options)


def residuals_dataset():
    # find easy dataset and exceptions
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

    model = TicTacToeNet(hidden_sizes=[None], input_size=rep_length)

    dataset = tttDataset(
        states_dict=fixed_states,
        board_rep_func=board_rep_func,
        len_rep=rep_length
    )

    train_to_perfection(
        model=model,
        dataset=dataset,
        save_checkpoint=False,
        max_epochs=1_000,
    )

    linear_boards, nonlinear_boards, router_boards = {}, {}, {}
    correct, num_wrong, total = 0, 0, 0
    for board_str, moves in all_states.items():
        total += 1

        board_rep = board_rep_func(board_str=board_str)
        board_tensor = torch.tensor(board_rep).float().unsqueeze(0)
        prediction = torch.argmax(model(board_tensor))

        if prediction in moves:
            correct += 1
            linear_boards[board_str] = [prediction.item()]
            router_boards[board_str] = [0]
        else:
            num_wrong += 1
            nonlinear_boards[board_str] = moves
            router_boards[board_str] = [1]

    print(f'Linear points: {correct} Nonlinear points: {num_wrong} total: {len(all_states)}')

    nonlinear_dataset = tttDataset(
        states_dict=nonlinear_boards,
        board_rep_func=board_rep_func,
        len_rep=rep_length
    )

    nn_friendly(
        all_states=nonlinear_boards,
        hidden_dim=15
    )

    # train_to_perfection(
    #     model=model_2,
    #     dataset=nonlinear_dataset,
    #     save_checkpoint=False,
    #     max_epochs=1_000,
    # )

    router_dataset = tttDataset(
        states_dict=router_boards,
        board_rep_func=board_rep_func,
        len_rep=rep_length
    )




if __name__ == '__main__':
    # svm_fit_per_output()

    # dec_tree_fit_per_output()

    # nn_friendly(hidden_dim=None, max_epochs=500)

    residuals_dataset()
