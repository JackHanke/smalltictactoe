import json
import random
import numpy as np
import torch
from tqdm import tqdm

from train import train_to_perfection
from data.dataset import tttDataset, alltttDataset
from models.nn import TicTacToeNet, routerNet
from data.reps import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def intervention_experiment():
    '''
    Examination for interventions of representation, number of layers, and illegal moves masking for smallest 100% models across `num_seeds` seeds
    
    RESULTS:
        layers: 1 replen: 18 masking: False | 765
        layers: 1 replen: 18 masking: True  | 597
        layers: 1 replen: 9  masking: False | 598
        layers: 1 replen: 9  masking: True  | 541   *BEST*
        layers: 2 replen: 18 masking: False | 1205
        layers: 2 replen: 18 masking: True  | 855
        layers: 2 replen: 9  masking: False | 870
        layers: 2 replen: 9  masking: True  | 693    
    '''

    with open("data/datasets/jsons/nn_friendly_dataset.json", "r") as file:
        states_dict = json.load(file)

    num_seeds = 4
    results_dict = {}
    for num_layers in [1,2]:
        for board_rep_func, rep_length in [(binary_board_rep, 18), (trinary_board_rep, 9)]:
            for do_illegal_move_masking in [True]:
                start_hidden_dim = 25
                best_params = float('inf')

                perfection_reached = True
                while perfection_reached:
                    print(f'Examining hidden_dim: {start_hidden_dim}...')

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
                            one_right_answer=False,
                            device=DEVICE,
                            save_checkpoint=False,
                            verbose=True,
                        )

                        if not seed_perfection_reached: 
                            perfection_reached = False
                            break
                        elif seed == num_seeds - 1:
                            perfection_reached = True

                    if perfection_reached:
                        best_params = num_params
                        start_hidden_dim -= 1

                checkpoint_path = f'models/checkpoints/intervention_experiment/nn_{best_params}_{hidden_sizes}_{num_layers}_{rep_length}_{do_illegal_move_masking}.pth'
                model.zero_grad() # zero grads for file size
                torch.save(model.state_dict(), checkpoint_path)
                # print(f'Model saved at: {checkpoint_path}')

                results_dict[f'{num_layers}_{rep_length}_{do_illegal_move_masking}'] = best_params
                print(f'> layers: {num_layers} replen: {rep_length} masking: {do_illegal_move_masking} | {best_params}')


def smallest_possible_experiment():
    '''
    Find best seed with best settings over `num_seeds`==30, single layer trinary rep with illegal move masking

    hidden_dim=24 seems to be the limit, 100 seeds tried with 23
    '''

    with open("data/datasets/jsons/nn_friendly_filtered_dataset.json", "r") as file:
        states_dict = json.load(file)

    start_hidden_dim = 11
    num_layers = 1
    best_params = float('inf')

    num_seeds = 100

    seed_found = True
    while seed_found:
        seed_found = False
        prog = tqdm(range(100, 100+num_seeds))
        for seed in prog:
            prog.set_description(f'Seed: {seed}')
            random.seed(seed)
            np.random.seed(seed)       
            torch.manual_seed(seed)

            board_rep_func, rep_length = (trinary_board_rep, 9)
            # board_rep_func, rep_length = (trinary_plus_sym_board_rep, 17)
            do_illegal_move_masking = True

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
            # dataset = alltttDataset(
            #     board_rep_func=board_rep_func,
            #     len_rep=rep_length,
            # )

            num_params = sum(p.numel() for p in model.parameters())
            print(f'Testing hidden_dim: {start_hidden_dim}, params {num_params}...')

            seed_perfection_reached, epoch, accuracy = train_to_perfection(
                model=model,
                dataset=dataset,
                max_epochs=40_000,
                weight_decay=0.0,
                one_right_answer=False,
                device=DEVICE,
                save_checkpoint=False,
                verbose=True,
            )

            if seed_perfection_reached: 
                print(f'Param {num_params} found seed: {seed}')
                seed_found = True
                checkpoint_path = f'models/checkpoints/smallest_possible_experiment/nn_{num_params}_{hidden_sizes}_{num_layers}_{rep_length}_{do_illegal_move_masking}_{seed}.pth'
                model.zero_grad() # zero grads for file size
                torch.save(model.state_dict(), checkpoint_path)
                break
            elif seed == num_seeds - 1:
                seed_found = False

        if seed_found:
            best_params = num_params
            start_hidden_dim -= 1

    print(f'Best params from seed hunt: {best_params}')

if __name__ == '__main__':

    # intervention_experiment()

    smallest_possible_experiment()