## function for how to represent the board and the move
    
def trinary_board_rep(board_str: str):
    ''' 9 trinary digits, 1 for X, -1 for O, 0 for empty '''
    bool_board = [0 for _ in range(9)]

    for idx, val in enumerate(board_str):
        if val == 'X':
            bool_board[idx] = 1

    for idx, val in enumerate(board_str):
        if val == 'O':
            bool_board[idx] = -1

    return bool_board

def binary_board_rep(board_str: str):
    ''' 18 binary digits, first 9 are position of X and second 9 are position of O '''
    bool_board = [0 for _ in range(18)]

    for idx, val in enumerate(board_str):
        if val == 'X':
            bool_board[idx] = 1

    for idx, val in enumerate(board_str):
        if val == 'O':
            bool_board[idx+9] = 1

    return bool_board

def one_hot_move_rep(move: int):
    ''' one hot encoding of 9 possible moves '''
    moves = [0 for _ in range(9)]
    moves[move] = 1
    return moves

def one_neg_one_move_rep(move: int):
    ''' one hot encoding of 9 possible moves '''
    moves = [-1 for _ in range(9)]
    moves[move] = 1
    return moves