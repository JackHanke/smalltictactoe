## thanks Gemini

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torchsummary import summary

from models.nn import TicTacToeNet
from train import train_to_perfection

def prune_train_loop(
        model,
        dataset,    
    ):

    max_epochs = 10_000

    gen = 0
    perfection_reached = True
    fraction_to_prune = 0.10
    patience = 3
    while True:
        gen += 1
        gens_since_no_improvement = 0

        parameters_to_tune = (
            (model.model[0], 'weight'),
            (model.model[0], 'bias'),
            (model.model[2], 'weight'),
            (model.model[2], 'bias'),
            (model.model[4], 'weight'),
            (model.model[4], 'bias'),
        )

        num_params = sum(p.numel() for p in model.parameters())
        
        if not perfection_reached or (num_params * fraction_to_prune < 1):
            gens_since_no_improvement += 1

            # print(model.model[0].weight_mask)
            
            # undo previous pruning NOTE so fucking hacky

            # print(model.model[0].weight_mask)

            if gens_since_no_improvement == patience or num_params * fraction_to_prune < 1: 
                model.model[0].weight_orig.data.copy_(current_best_state_dict['model.0.weight_orig'])
                model.model[0].bias_orig.data.copy_(current_best_state_dict['model.0.bias_orig'])
                model.model[0].weight_mask.data.copy_(current_best_state_dict['model.0.weight_mask'])
                model.model[0].bias_mask.data.copy_(current_best_state_dict['model.0.bias_mask'])

                model.model[2].weight_orig.data.copy_(current_best_state_dict['model.2.weight_orig'])
                model.model[2].bias_orig.data.copy_(current_best_state_dict['model.2.bias_orig'])
                model.model[2].weight_mask.data.copy_(current_best_state_dict['model.2.weight_mask'])
                model.model[2].bias_mask.data.copy_(current_best_state_dict['model.2.bias_mask'])

                # make pruning permanent 
                for (param, param_type) in parameters_to_tune:
                    prune.remove(param, param_type)

                checkpoint_path = f'models/checkpoints/nn_gen{gen}_nonz{nonzero_params}_{model.hidden_sizes}.pth'
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

        nonzero_params = torch.sum(model.model[0].weight != 0) + \
            torch.sum(model.model[0].bias != 0) + \
            torch.sum(model.model[2].weight != 0) + \
            torch.sum(model.model[2].bias != 0) + \
            torch.sum(model.model[4].weight != 0) + \
            torch.sum(model.model[4].bias != 0)
        print(f'\nGen {gen} non-zero params testing: {nonzero_params}\n')

        perfection_reached, _, _ = train_to_perfection(
            model=model,
            dataset=dataset,
            max_epochs=max_epochs,
            save_checkpoint=False,
            name=f'_gen{gen}_nonz{nonzero_params}'
        )

        if perfection_reached:
            # save current best perfect state dict
            current_best_state_dict = model.state_dict().copy()

