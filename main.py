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

    num_layers = 1
    HIDDEN_DIMS = [5 for _ in range(num_layers)]
    rep_length = 9
    board_rep_fn = trinary_board_rep
    model = TicTacToeNet(
        hidden_sizes=HIDDEN_DIMS,
        input_size=rep_length
    ).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {num_params}')

    PATH = 'models/checkpoints/smallest_possible_experiment/nn_104_[5]_1_9_True_5.pth'
    model.load_state_dict(torch.load(PATH, weights_only=True))

    DATA_PATH = "data/datasets/jsons/non_block_or_win_filtered.json"
    print(f'DATA PATH: {DATA_PATH}')
    data_tensor, moves_mask = data_json_to_tensor(
        data_json_path=DATA_PATH,
        board_rep_fn=board_rep_fn,
    )

    # print(f'Hidden dims: {HIDDEN_DIMS}')

    # train_to_perfection(
    #     model=model,
    #     dataset=dataset,
    #     max_epochs=10_000,
    #     weight_decay=0.0,
    #     one_right_answer=False,
    #     learning_rate=1e-2,
    #     device=DEVICE,
    #     verbose=True
    # )

    prune_train_loop(
        model=model,
        data_tensor=data_tensor,
        moves_mask=moves_mask,
        device=DEVICE,
        max_epochs=50_000,
    )
