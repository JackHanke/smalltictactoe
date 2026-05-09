import json
import torch
from torchsummary import summary

from train import train_to_perfection
from data.dataset import tttDataset
from models.nn import TicTacToeNet
from models.difflogic.difflogic.difflogic import LogicLayer, GroupSum
from data.reps import *
from prune import prune_train_loop

if __name__ == "__main__":
    HIDDEN_DIMS = [32]
    rep_length = 9
    # board_rep_func = binary_board_rep
    board_rep_func = trinary_board_rep
    model = TicTacToeNet(hidden_sizes=HIDDEN_DIMS, input_size=rep_length)

    PATH = '/Users/jack/vault/software/smalltictactoe/models/checkpoints/nn_[9, 32, 9]_617_2026-05-03-13:18:18.pth'
    model.load_state_dict(torch.load(PATH, weights_only=True))

    # summary(model)

    # HIDDEN_DIM = 2_700
    # model = torch.nn.Sequential(
    #     torch.nn.Flatten(),
    #     LogicLayer(in_dim=18, out_dim=HIDDEN_DIM, device='cpu'),
    #     LogicLayer(in_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM, device='cpu'),
    #     LogicLayer(in_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM, device='cpu'),
    #     LogicLayer(in_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM, device='cpu'),
    #     GroupSum(k=9, tau=(HIDDEN_DIM//9))
    # )

    # with open("data/seed_options.json", "r") as file:
    #     states = json.load(file)

    # states_dict = {}
    # for key, value in states.items():
    #     val = value[0]
    #     states_dict[key] = [val]

    # with open(f'data/{rep_length}_example_dataset.json', 'w') as fp:
    #     json.dump(states_dict, fp)

    with open(f"data/{rep_length}_example_dataset.json", "r") as file:
        states_dict = json.load(file)

    dataset = tttDataset(
        states_dict=states_dict,
        board_rep_func=board_rep_func,
        len_rep=rep_length,
    )    

    # print(f'Hidden dims: {HIDDEN_DIMS}')

    # train_to_perfection(
    #     model=model,
    #     dataset=dataset,
    #     max_epochs=20_000,
    #     weight_decay=0.0,
    # )

    # prune_train_loop(
    #     model=model,
    #     dataset=dataset,
    # )


