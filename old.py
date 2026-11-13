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
    #     input_size=8,
    #     hidden_sizes=[7],
    #     output_size=1
    # ).to(DEVICE)
    # PATH = 'models/checkpoints/nn_cornerclass_[8, 7, 1]_71_2026-10-30-11:54:07.pth'
    # model.load_state_dict(torch.load(PATH, weights_only=True, map_location=DEVICE))

    # model = tttclassifer(
    #     input_size=9,
    #     hidden_sizes=[3],
    #     output_size=1
    # ).to(DEVICE)
    # PATH = 'models/checkpoints/nn_edgeclass_[9, 3, 1]_34_2026-11-12-17:53:51.pth'
    # model.load_state_dict(torch.load(PATH, weights_only=True, map_location=DEVICE))

    model = tttclassifer(
        input_size=9,
        hidden_sizes=[6],
        output_size=1
    ).to(DEVICE)
    PATH = '/Users/jack/vault/software/smalltictactoe/models/checkpoints/nn_cornerclass_[9, 6, 1]_67_2026-11-12-23:43:50.pth'
    model.load_state_dict(torch.load(PATH, weights_only=True, map_location=DEVICE))

    # prune_train_loop(
    #     model=model,
    #     data_tensor=data_tensor,
    #     moves_mask=moves_mask,
    #     device=DEVICE,
    #     max_epochs=100_000,
    # )

    # DATA_PATH = 'data/datasets/jsons/other.json'
    # board_rep_fn, rep_length = (trinary_board_rep, 9)
    # data_tensor, moves_mask = data_json_to_tensor(
    #     data_json_path=DATA_PATH,
    #     board_rep_fn=board_rep_fn,
    # )

    num_moves_to_resolve = 15
    inclusion_code = [0 for _ in range(num_moves_to_resolve)]
    state_matrix_0, y_0, state_matrix_1, y_1, _ = make_data(inclusion_code)

    # if type(columns) is tuple: columns = [columns]
    # for col_tup in columns:
    # if move == 0:
    state_matrix_0 = augment_state_matrix(state_matrix_0, (4,6))
    # if move == 1:
    state_matrix_1 = augment_state_matrix(state_matrix_1, (0,4))

    # move = 0
    move = 0
    prune_simple_train_loop(
        model=model,
        data_tensor=state_matrix_0,
        device=DEVICE,
        move=move,
        max_epochs=50_000,
        name=f'{move}'
    )
