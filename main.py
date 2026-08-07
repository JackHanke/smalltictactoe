import json
import random
import numpy as np
import torch

from train import train_to_perfection
from data.dataset import tttDataset, alltttDataset
from models.nn import TicTacToeNet, routerNet
from data.reps import *
from prune import prune_train_loop
from data.game import generate_states_from_root_board
from experiments import *

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # smallest_possible_experiment()

    HIDDEN_DIMS = [16,16]
    PATH = '/home/jack/vault/software/smalltictactoe/models/checkpoints/smallest_possible_experiment/nn_585_[16, 16]_2_9_True_1.pth'
    rep_length = 9
    board_rep_func = trinary_board_rep
    model = TicTacToeNet(
        hidden_sizes=HIDDEN_DIMS,
        input_size=rep_length
    ).to(DEVICE)
    # model = routerNet(hidden_sizes=HIDDEN_DIMS, input_size=rep_length)

    model.load_state_dict(torch.load(PATH, weights_only=True))

    with open("data/_9_seed_options.json", "r") as file:
        states_dict = json.load(file)


    # states_dict = {}
    # for key, value in states.items():
    #     val = value[0]
    #     states_dict[key] = [val]

    # with open(f'data/example_dataset.json', 'w') as fp:
    #     json.dump(states_dict, fp)

    # with open(f"data/9_example_dataset.json", "r") as file:
    #     states_dict = json.load(file)

    # with open(f"data/router_dataset.json", "r") as file:
    #     states_dict = json.load(file)

    # with open(f"/Users/jack/vault/software/smalltictactoe/_9_seed_options.json", "r") as file:
    #     states_dict = json.load(file)

    # dataset = alltttDataset(
    #     states_dict=states_dict,
    #     board_rep_func=board_rep_func,
    #     len_rep=rep_length,
    # )    
    dataset = tttDataset(
        states_dict=states_dict,
        board_rep_func=board_rep_func,
        len_rep=rep_length,
    )    

    # print(f'Hidden dims: {HIDDEN_DIMS}')

    # train_to_perfection(
    #     model=model,
    #     dataset=dataset,
    #     max_epochs=10_000,
    #     weight_decay=0.0,
    #     one_right_answer=True,
    #     device=DEVICE,
    # )

    prune_train_loop(
        model=model,
        dataset=dataset,
        device=DEVICE,
    )
