# thanks Gemini

import pandas as pd
import torch

def encode_board(board_str):
    mapping = {'X': 1, 'O': -1, ' ': 0}
    return [mapping[char] for char in board_str]

def get_data(path: str):
    df = pd.read_csv(path)
    X_data = torch.tensor([encode_board(s) for s in df['board_state']], dtype=torch.float32)
    y_data = torch.tensor(df['best_move_index'].values, dtype=torch.long)
    return X_data, y_data

if __name__ == '__main__':
    # count the total number of minimax datasets possible
    PATH = 'data/datasets/tictactoe_best_moves_all.csv'
    df = pd.read_csv(PATH)
    prod = 1
    for best_moves_str in df['best_move_index']:
        prod *= (len(best_moves_str)-2)
    print(f'The total number of datasets: {prod}')

