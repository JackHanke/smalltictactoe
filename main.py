import json
import torch
from torchsummary import summary

from train import train_to_perfection
from data.dataset import tttDataset
from models.nn import TicTacToeNet
from models.difflogic.difflogic.difflogic import LogicLayer, GroupSum
from data.reps import *

if __name__ == "__main__":
    HIDDEN_DIMS = [25]
    rep_length = 18
    board_rep_func = binary_board_rep
    model = TicTacToeNet(hidden_sizes=HIDDEN_DIMS, input_size=rep_length)

    # PATH = '/Users/jack/vault/software/smalltictactoe/models/checkpoints/nn_22_625_2026-04-26-22:35:01.pth'
    # model.load_state_dict(torch.load(PATH, weights_only=True))

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

    with open("seed_options.json", "r") as file:
        states = json.load(file)

    states_dict = {}
    for key, value in states.items():
        val = value[0]
        states_dict[key] = [val]

    with open('example_dataset.json', 'w') as fp:
        json.dump(states_dict, fp)

    # dataset = tttDataset(
    #     states_dict=states_dict,
    #     board_rep_func=board_rep_func,
    #     len_rep=rep_length,
    # )    

    # print(f'Hidden dims: {HIDDEN_DIMS}')

    # train_to_perfection(
    #     model=model,
    #     dataset=dataset,
    #     max_epochs=5_000
    # )
