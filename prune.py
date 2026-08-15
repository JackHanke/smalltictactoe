## thanks Gemini

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from models.nn import TicTacToeNet
from train import train_to_perfection

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


