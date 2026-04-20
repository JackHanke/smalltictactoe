# thanks Gemini kinda
import csv
from game import check_winner, get_best_move

def nn_rep(board):
    ''' 9 trinary digits, 1 for X, -1 for O, 0 for empty '''
    bool_board = ['0' for _ in range(9)]

    for idx, val in enumerate(board):
        if val == 'X':
            bool_board[idx] = '1'

    for idx, val in enumerate(board):
        if val == 'O':
            bool_board[idx] = '-1'

    return bool_board

def bin_rep(board):
    ''' 18 binary digits, first 9 are position of X and second 9 are position of O '''
    bool_board = ['0' for _ in range(18)]

    for idx, val in enumerate(board):
        if val == 'X':
            bool_board[idx] = '1'

    for idx, val in enumerate(board):
        if val == 'O':
            bool_board[idx+9] = '1'

    return bool_board

def one_hot_move(move):
    ''' one hot encoding of 9 possible moves '''
    moves = ['0' for _ in range(9)]
    moves[move] = '1'
    return moves


all_states = {}
def generate_states(board, player):
    # Generate all legal reachable states
    board_str = "".join(board)
    if board_str in all_states or check_winner(board):
        return
    
    best_move = get_best_move(board, player)
    all_states[board_str] = (player, best_move)
    
    for i in range(9):
        if board[i] == ' ':
            board[i] = player
            generate_states(board, 'O' if player == 'X' else 'X')
            board[i] = ' '
    
    return all_states

def create_dataset(path: str):
    generate_states([' '] * 9, 'X')

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'{i}' for i in range(18+9)])
        for state, (player, best_move) in sorted(all_states.items()):
            ## board representation
            board_rep = bin_rep(board=state)
            # board_rep = nn_rep(board=state)

            ## move representation
            move_rep = one_hot_move(move=int(best_move))
            # move_rep = [int(best_move)]

            ## write
            writer.writerow(board_rep+move_rep)

def create_all_dataset(path: str):
    ''' create dataset  '''
    all_states = generate_states([' '] * 9, 'X')
    
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['board_state', 'player_turn', 'best_move_index'])
        for state, info in sorted(all_states.items()):

            writer.writerow([state, info[0], "'"+info[1]+"'"])

    print(f"Dataset generated with {len(all_states)} unique board states.")

def create_espresso_dataset(path: str):
    ''' dataset for binary circuit minimization with espresso '''
    all_states = generate_states([' '] * 9, 'X')
    
    with open(path, "w") as f:
        f.write('.i 13\n')
        f.write('.o 4\n')
        for board_num, (state, (player, best_move)) in enumerate(sorted(all_states.items())):
            board_str = ''.join(bin_rep(board=state))
            move_str = ''.join(one_hot_move(move=int(best_move)))
            board_num_b = format(board_num, '013b')
            move_num_b = format(int(best_move), '04b')
            # f.write(f'{board_str} {move_str}\n')
            # f.write(f'{board_num_b} {move_str}\n')
            f.write(f'{board_num_b} {move_num_b}\n')

if __name__ == '__main__':
    PATH = f'data/datasets/ttt_binary.csv'

    create_dataset(path=PATH)
    # create_dataset(path=PATH)
    # create_all_dataset(path=PATH)
    # create_espresso_dataset()