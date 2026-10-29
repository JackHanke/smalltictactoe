## thanks Gemini

import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from models.nn import TicTacToeNet
from train import train_to_perfection
from experiments import simple_train_loop
from main import filter_feature_state_matrix_among_0_1

def prune_train_loop(
        model,
        data_tensor,
        moves_mask,
        device,
        max_epochs:int = 10_000,
    ):

    gen = 0
    perfection_reached = True
    fraction_to_prune = 0.05
    patience = 4
    gens_since_no_improvement = 0
    while True:
        gen += 1

        parameters_to_tune = []
        for i in range(len(model.hidden_sizes)-1):
            parameters_to_tune.append((model.model[2*i], 'weight'))
            parameters_to_tune.append((model.model[2*i], 'bias'))

        num_params = sum(p.numel() for p in model.parameters())
        
        if not perfection_reached or (num_params * fraction_to_prune < 1):
            # undo previous pruning NOTE so fucking hacky
            gens_since_no_improvement += 1

            for i in range(len(model.hidden_sizes)-1):
                model.model[2*i].weight_orig.data.copy_(current_best_state_dict[f'model.{2*i}.weight_orig'])
                model.model[2*i].bias_orig.data.copy_(current_best_state_dict[f'model.{2*i}.bias_orig'])
                model.model[2*i].weight_mask.data.copy_(current_best_state_dict[f'model.{2*i}.weight_mask'])
                model.model[2*i].bias_mask.data.copy_(current_best_state_dict[f'model.{2*i}.bias_mask'])
            
            if gens_since_no_improvement == patience or num_params * fraction_to_prune < 1: 

                # make pruning permanent 
                for (param, param_type) in parameters_to_tune:
                    prune.remove(param, param_type)

                checkpoint_path = f'models/checkpoints/pruning/nn_nonz_{nonzero_params}_{num_params}_{model.hidden_sizes}.pth'
                model.zero_grad() # zero grads for file size
                torch.save(model.state_dict(), checkpoint_path)
                return

            fraction_to_prune *= 0.5
        else:
            gens_since_no_improvement = 0


        prune.global_unstructured(
            parameters=parameters_to_tune,
            pruning_method=prune.L1Unstructured,
            amount=fraction_to_prune,
        )

        # print(f'-----{gen}----')
        # for key, val in model.state_dict().items():
        #     print(f'{key} : \n{val.shape}')
        # input(model.model[0].weight_mask)

        nonzero_params = 0
        for i in range(len(model.hidden_sizes)-1):
            nonzero_params += torch.sum(model.model[2*i].weight != 0)
            nonzero_params += torch.sum(model.model[2*i].bias != 0)

        print(f'\nGen {gen} non-zero params testing: {nonzero_params}/{num_params}')

        perfection_reached, _, _ = train_to_perfection(
            model=model,
            data_tensor=data_tensor,
            moves_mask=moves_mask,
            max_epochs=max_epochs,
            save_checkpoint=False,
            one_right_answer=False,
            name=f'_gen{gen}_nonz{nonzero_params}',
            device=device,
            verbose=True,
        )

        if perfection_reached:
            # save current best perfect state dict
            current_best_state_dict = model.state_dict().copy()
            gens_since_no_improvement = 0

def prune_simple_train_loop(
        model,
        data_tensor,
        device,
        max_epochs:int = 10_000,
    ):

    gen = 0
    perfection_reached = True
    fraction_to_prune = 0.05
    patience = 4
    gens_since_no_improvement = 0

    DATA_PATH = 'data/datasets/jsons/other.json'
    with open(DATA_PATH, 'r') as f: states = json.load(f)
    inclusion_code = [1 for _ in range(100)]
    state_matrix_0, y_0, state_matrix_1, y_1 = filter_feature_state_matrix_among_0_1(states, data_tensor.numpy(), inclusion_code)
    
    state_matrix_0 = torch.from_numpy(state_matrix_0).to(device, non_blocking=True)
    state_matrix_0 = state_matrix_0[:, [1,2,3,4,5,6,7,8]]
    y_0 = torch.tensor(y_0).to(device, non_blocking=True)

    state_matrix_1 = torch.from_numpy(state_matrix_1).to(device, non_blocking=True)
    state_matrix_1 = state_matrix_1[:, [0,2,3,4,5,6,7,8]]
    y_1 = torch.tensor(y_1).to(device, non_blocking=True)


    while True:
        gen += 1

        parameters_to_tune = []
        for i in range(len(model.hidden_sizes)-1):
            parameters_to_tune.append((model.model[2*i], 'weight'))
            parameters_to_tune.append((model.model[2*i], 'bias'))

        num_params = sum(p.numel() for p in model.parameters())
        
        if not perfection_reached or (num_params * fraction_to_prune < 1):
            # undo previous pruning NOTE so fucking hacky
            gens_since_no_improvement += 1

            for i in range(len(model.hidden_sizes)-1):
                model.model[2*i].weight_orig.data.copy_(current_best_state_dict[f'model.{2*i}.weight_orig'])
                model.model[2*i].bias_orig.data.copy_(current_best_state_dict[f'model.{2*i}.bias_orig'])
                model.model[2*i].weight_mask.data.copy_(current_best_state_dict[f'model.{2*i}.weight_mask'])
                model.model[2*i].bias_mask.data.copy_(current_best_state_dict[f'model.{2*i}.bias_mask'])
            
            if gens_since_no_improvement == patience or num_params * fraction_to_prune < 1: 

                # make pruning permanent 
                for (param, param_type) in parameters_to_tune:
                    prune.remove(param, param_type)

                checkpoint_path = f'models/checkpoints/pruning/nn_nonz_{nonzero_params}_{num_params}_{model.hidden_sizes}.pth'
                model.zero_grad() # zero grads for file size
                torch.save(model.state_dict(), checkpoint_path)
                return

            fraction_to_prune *= 0.5
        else:
            gens_since_no_improvement = 0


        prune.global_unstructured(
            parameters=parameters_to_tune,
            pruning_method=prune.L1Unstructured,
            amount=fraction_to_prune,
        )

        # print(f'-----{gen}----')
        # for key, val in model.state_dict().items():
        #     print(f'{key} : \n{val.shape}')
        # input(model.model[0].weight_mask)

        nonzero_params = 0
        for i in range(len(model.hidden_sizes)-1):
            nonzero_params += torch.sum(model.model[2*i].weight != 0)
            nonzero_params += torch.sum(model.model[2*i].bias != 0)

        print(f'\nGen {gen} non-zero params testing: {nonzero_params}/{num_params}')

        if move == 0:
            perfection_reached, _, _ = simple_train_loop(
                model=model,
                state_tensor=state_matrix_0,
                y=y_0,
                max_epochs=100_000,
                save_checkpoint = False,
                name='cornerclass',
            )

        elif move == 1:
            perfection_reached, _, _ = simple_train_loop(
                model=model,
                state_tensor=state_matrix_1,
                y=y_1,
                max_epochs=100_000,
                save_checkpoint = False,
                name='edgeclass',
            )

        if perfection_reached:
            # save current best perfect state dict
            current_best_state_dict = model.state_dict().copy()
            gens_since_no_improvement = 0
