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

def trinary_plus_sym_board_rep(board_str: str):
    bool_board = trinary_board_rep(board_str=board_str)

    c1, e1, c2, e2, c,  e3, c3, e4, c4 = bool_board
    bool_board.extend([
        # c,
        c1 + c2 + c3 + c4,
        c2*c3 + c1*c4,
        e1*e2 + e1*e3 + e2*e4 + e3*e4,
        e1**2 + e2**2 + e3**2 + e4**2,
        c1*c2 + c1*c3 + c2*c4 + c3*c4,
        c1*c2*e4 + c2*c4*e2 + c3*c4*e1 + c1*c3*e3,
        c1*e1*e2 + c2*e1*e3 + c4*e3*e4 + c3*e2*e4,
        c1*e1 + c1*e2 + c2*e1 + c2*e3 + c4*e3 + c4*e4 + c3*e2 + c3*e4,
     ])
    return bool_board

def trinary_plus_handcrafted_board_rep(board_str: str):
    bool_board = trinary_board_rep(board_str=board_str)

    c1, e1, c2, e2, c,  e3, c3, e4, c4 = bool_board
    bool_board.extend([
        c1*e2,
        c1*c3,
        e2*c3,
        e1*c,
        e1*e4,
        c*e4,
        c2*e3,
        c2*c4,
        e3*c4,

        c1*e1,
        c1*c2,
        e1*c2,
        e2*c,
        e2*e3,
        c*e3,
        c3*e4,
        c3*c4,
        e4*c4,

        c1*c,
        c1*c4,
        c*c4,
        c2*c,
        c2*c3,
        c*c3,
     ])
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