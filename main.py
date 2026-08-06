import json
import random
import numpy as np
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

    with open("data/_9_seed_options.json", "r") as file:
        states_dict = json.load(file)

    num_seeds = 5   
    results_dict = {}
    for num_layers in [1,2]:
        for board_rep_func, rep_length in [(binary_board_rep, 18), (trinary_board_rep, 9)]:
            for do_illegal_move_masking in [True]:
                start_hidden_dim = 35
                best_params = float('inf')

                perfection_reached = True
                while perfection_reached:

                    for seed in range(num_seeds):
                        random.seed(seed)
                        np.random.seed(seed)       
                        torch.manual_seed(seed)

                        hidden_sizes = [start_hidden_dim for _ in range(num_layers)]
                        model = TicTacToeNet(
                            hidden_sizes=hidden_sizes,
                            input_size=rep_length,
                            do_illegal_move_masking=do_illegal_move_masking,
                        ).to(DEVICE)

                        dataset = tttDataset(
                            states_dict=states_dict,
                            board_rep_func=board_rep_func,
                            len_rep=rep_length,
                        )   

                        num_params = sum(p.numel() for p in model.parameters())

                        seed_perfection_reached, epoch, accuracy = train_to_perfection(
                            model=model,
                            dataset=dataset,
                            max_epochs=10_000,
                            weight_decay=0.0,
                            one_right_answer=True,
                            device=DEVICE,
                            save_checkpoint=False,
                        )

                        if not seed_perfection_reached: 
                            perfection_reached = False
                            break
                        elif seed == num_seeds - 1:
                            perfection_reached = True

                    if perfection_reached:
                        best_params = num_params
                        start_hidden_dim -= 1

                checkpoint_path = f'models/checkpoints/experiment/nn_{best_params}_{hidden_sizes}_{num_layers}_{rep_length}_{do_illegal_move_masking}.pth'
                model.zero_grad() # zero grads for file size
                torch.save(model.state_dict(), checkpoint_path)
                # print(f'Model saved at: {checkpoint_path}')

                results_dict[f'{num_layers}_{rep_length}_{do_illegal_move_masking}'] = best_params
                print(f'> layers: {num_layers} replen: {rep_length} masking: {do_illegal_move_masking} | {best_params}')


    '''
    RESULTS:::
    > layers: 1 replen: 18 masking: False | 765
    > layers: 1 replen: 18 masking: True | 597
    > layers: 1 replen: 9 masking: False | 598
    > layers: 1 replen: 9 masking: True | 541
    > layers: 2 replen: 18 masking: False | 1205
    > layers: 2 replen: 18 masking: True | 855
    > layers: 2 replen: 9 masking: False | 870
    > layers: 2 replen: 9 masking: True | 693    
    '''


    # HIDDEN_DIMS = [2]
    # rep_length = 9
    # # board_rep_func = binary_board_rep
    # board_rep_func = trinary_board_rep
    # model = TicTacToeNet(hidden_sizes=HIDDEN_DIMS, input_size=rep_length).to(DEVICE)
    # model = routerNet(hidden_sizes=HIDDEN_DIMS, input_size=rep_length)

    # PATH = 'models/checkpoints/nn_[9, 30, 9]_579_2026-05-12-22:34:28.pth'
    # model.load_state_dict(torch.load(PATH, weights_only=True))

    # with open("data/_9_seed_options.json", "r") as file:
    #     states_dict = json.load(file)


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
    # dataset = tttDataset(
    #     states_dict=states_dict,
    #     board_rep_func=board_rep_func,
    #     len_rep=rep_length,
    # )    

    # print(f'Hidden dims: {HIDDEN_DIMS}')

    # train_to_perfection(
    #     model=model,
    #     dataset=dataset,
    #     max_epochs=10_000,
    #     weight_decay=0.0,
    #     one_right_answer=True,
    #     device=DEVICE,
    # )

    # prune_train_loop(
    #     model=model,
    #     dataset=dataset,
    #     device=DEVICE,
    # )
