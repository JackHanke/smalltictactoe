import pandas as pd
import torch

def encode_board(board_str):
    mapping = {'X': 1, 'O': -1, ' ': 0}
    return [mapping[char] for char in board_str]

def get_data():
    df = pd.read_csv('tictactoe_best_moves.csv')
    X_data = torch.tensor([encode_board(s) for s in df['board_state']], dtype=torch.float32)
    y_data = torch.tensor(df['best_move_index'].values, dtype=torch.long)
    return X_data, y_data
