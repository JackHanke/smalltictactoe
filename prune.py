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

    gen = 0

    parameters_to_tune = (
        (model.model[0], 'weight'),
        (model.model[2], 'weight'),
    )

    prune.global_unstructured(
        parameters=parameters_to_tune,
        pruning_method=prune.L1Unstructured,
        amount=0.01,
    )

    nonzero_params = torch.sum(model.model[0] != 0) + torch.sum(model.model[2] != 0)

    train_to_perfection(
        model=model,
        dataset=dataset,
        max_epochs=None,
        name=f'_gen{gen}_nonz{nonzero_params}'
    )


