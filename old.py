import json
import random
import numpy as np
import torch

from train import train_to_perfection
from data.dataset import tttDataset, alltttDataset
from models.nn import TicTacToeNet, routerNet
from data.reps import *
from prune import *
from data.game import generate_states_from_root_board
from experiments import *


if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # seed = 10
    # random.seed(seed)
    # np.random.seed(seed)       
    # torch.manual_seed(seed)

    # num_layers = 1
    # HIDDEN_DIMS = [9 for _ in range(num_layers)]
    # rep_length = 9
    # board_rep_fn = trinary_board_rep
    # model = TicTacToeNet(
    #     hidden_sizes=HIDDEN_DIMS,
    #     input_size=rep_length,
    #     do_illegal_move_masking=True,
    # ).to(DEVICE)
    # num_params = sum(p.numel() for p in model.parameters())
    # print(f'Model parameters: {num_params}')

    # DATA_PATH = "data/datasets/jsons/other.json"
    # print(f'DATA PATH: {DATA_PATH}')
    # data_tensor, moves_mask = data_json_to_tensor(
    #     data_json_path=DATA_PATH,
    #     board_rep_fn=board_rep_fn,
    # )

    # print(f'Hidden dims: {HIDDEN_DIMS}')

    # train_to_perfection(
    #     model=model,
    #     data_tensor=data_tensor,
    #     moves_mask=moves_mask,
    #     max_epochs=25_000,
    #     weight_decay=0.0,
    #     one_right_answer=False,
    #     device=DEVICE,
    #     save_checkpoint=True,
    #     verbose=True,
    #     seed = seed,
    #     name='other'
    # )

    # model = tttclassifer(
    #     hidden_sizes=[7],
    # ).to(DEVICE)
    # PATH = 'models/checkpoints/nn_cornerclass_[8, 7, 2]_79_2026-10-28-21:19:09.pth'
    # model.load_state_dict(torch.load(PATH, weights_only=True))

    model = tttclassifer(
        input_size=8,
        hidden_sizes=[7],
    ).to(DEVICE)
    PATH = 'models/checkpoints/nn_cornerclass_[8, 7, 2]_79_2026-10-28-21:19:09.pth'
    model.load_state_dict(torch.load(PATH, weights_only=True))

    # prune_train_loop(
    #     model=model,
    #     data_tensor=data_tensor,
    #     moves_mask=moves_mask,
    #     device=DEVICE,
    #     max_epochs=100_000,
    # )

    DATA_PATH = 'data/datasets/jsons/other.json'
    board_rep_fn, rep_length = (trinary_board_rep, 9)
    data_tensor, moves_mask = data_json_to_tensor(
        data_json_path=DATA_PATH,
        board_rep_fn=board_rep_fn,
    )

    move = 0
    move = 1
    prune_simple_train_loop(
        model=model,
        data_tensor=data_tensor,
        device=DEVICE,
        max_epochs=100_000,
    )
