import json
import torch

from train import train_to_perfection
from data.dataset import tttDataset, alltttDataset
from models.nn import TicTacToeNet, routerNet
# from models.difflogic.difflogic.difflogic import LogicLayer, GroupSum

from data.reps import *
from prune import prune_train_loop
from data.game import generate_states_from_root_board

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    HIDDEN_DIMS = [30]
    rep_length = 9
    # board_rep_func = binary_board_rep
    board_rep_func = trinary_board_rep
    model = TicTacToeNet(hidden_sizes=HIDDEN_DIMS, input_size=rep_length).to(DEVICE)
    # model = routerNet(hidden_sizes=HIDDEN_DIMS, input_size=rep_length)

    PATH = 'models/checkpoints/nn_[9, 30, 9]_579_2026-05-12-22:34:28.pth'
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
    #     max_epochs=25_000,
    #     weight_decay=0.0,
    #     one_right_answer=True,
    #     device=DEVICE,
    # )

    prune_train_loop(
        model=model,
        dataset=dataset,
        device=DEVICE,
    )
