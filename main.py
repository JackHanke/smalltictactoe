import torch
from torchsummary import summary

from train import train
from data.dataset import tttDataset
from models.nn import TicTacToeNet
from models.difflogic.difflogic.difflogic import LogicLayer, GroupSum

if __name__ == "__main__":
    HIDDEN_DIM = 2_700
    # model = TicTacToeNet(hidden_size=HIDDEN_DIM)
    # summary(model)

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        LogicLayer(in_dim=18, out_dim=HIDDEN_DIM, device='cpu'),
        LogicLayer(in_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM, device='cpu'),
        LogicLayer(in_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM, device='cpu'),
        LogicLayer(in_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM, device='cpu'),
        GroupSum(k=9, tau=(HIDDEN_DIM//9))
    )

    print(f'Hidden dim: {HIDDEN_DIM}')

    train(
        model=model,
        path='data/datasets/ttt_best_bin.csv',
        # path='data/datasets/ttt_nn.csv',
        len_rep=18,
    )
