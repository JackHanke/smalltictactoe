import random
import json

def check_winner(board):
    win_configs = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # Cols
        [0, 4, 8], [2, 4, 6]             # Diagonals
    ]
    for config in win_configs:
        if board[config[0]] == board[config[1]] == board[config[2]] != ' ':
            return board[config[0]]
    if ' ' not in board:
        return 'Tie'
    return None

def check_one_away_winner(board, player):
    result = None
    for i in range(9):
        if board[i] == ' ':
            board[i] = player
            winner = check_winner(board)
            if winner is not None or winner != 'Tie':
                result = winner
            board[i] = ' '
    return result

def minimax(board, player):
    winner = check_winner(board)
    if winner == 'X': return 1, [1,0,0]
    if winner == 'O': return -1, [0,0,1]
    if winner == 'Tie': return 0, [0,1,0]

    scores = []
    line_count = [0,0,0]
    
    for i in range(9):
        if board[i] == ' ':
            board[i] = player
            score, counts = minimax(board, 'O' if player == 'X' else 'X')
            line_count[0] += counts[0]
            line_count[1] += counts[1]
            line_count[2] += counts[2]
            scores.append(score)
            board[i] = ' '

    if player == 'X':
        return max(scores), line_count
    else:
        return min(scores), line_count

def get_best_move(
        board, 
        player: str,
    ):
    """
    Finds all moves sharing the best evaluation and returns one 
    based on the provided seed.
    """
    best_score = -float('inf') if player == 'X' else float('inf')
    best_moves = []
    
    # 1. Evaluate all legal moves
    for i in range(9):
        if board[i] == ' ':
            board[i] = player
            # We only care about the score (index 0) for move selection
            score, _ = minimax(board, 'O' if player == 'X' else 'X')
            board[i] = ' '
            
            # 2. Update best_moves list based on score
            if (player == 'X' and score > best_score) or (player == 'O' and score < best_score):
                best_score = score
                best_moves = [i] # Reset list with the new best move
            elif score == best_score:
                best_moves.append(i) # Add to the list of tied best moves

    return best_moves

def generate_states_helper(
        states: dict,
        board, 
        player: str, 
    ):
    # Generate all legal reachable states from `board`
    board_str = "".join(board)
    if board_str in states or check_winner(board) or check_one_away_winner(board, player):
        return states
    
    best_moves = get_best_move(
        board=board,
        player=player,
    )
    states[board_str] = best_moves
    
    for i in range(9):
        if board[i] == ' ':
            board[i] = player
            generate_states_helper(
                states=states,
                board=board, 
                player = 'O' if player == 'X' else 'X',
            )
            board[i] = ' '
    
    return states

def generate_states_from_root_board(
        board, 
        player: str, 
        seed: list[int] = None, 
    ):
    # 
    states = {}
    states = generate_states_helper(
        states=states,
        board=board,
        player=player,
    )

    if seed is None: return states

    seeded_states = {}
    for board_idx, (board, moves) in enumerate(states.items()):
        seeded_states[board] = [states[board][seed[board_idx]]]

    return seeded_states

if __name__ == '__main__':

    # with open("seed_options.json", "r") as file:
    #     new_options = json.load(file)

    # print(new_options)



    all_states = generate_states_from_root_board(
        board=[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        player='X',
    )
    # with open('all_states.json', 'w') as fp:
    #     json.dump(all_states, fp)
    # states_0 = generate_states_from_root_board(
    #     board=['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    #     player='O',
    # )
    # states_0[' '*9] = ('X', [0])
    # # states_1 = generate_states_from_root_board(
    # #     board=[' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    # #     player='O',
    # # )
    # # states_1[' '*9] = ('X', [1])
    # # states_4 = generate_states_from_root_board(
    # #     board=[' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' '],
    # #     player='O',
    # # )
    # # states_4[' '*9] = ('X', [4])

    hist = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

    # options = []
    
    for key, value in all_states.items(): 
        print(f'{key}: {value}')
        hist[len(value)] += 1
    #     options.append(value)

    # # print(f'options: {options}')

    # print(f'total points: {len(all_states)}')

    for key, value in hist.items(): print(f'hist {key}: {value}')
    print(f'Total datapoints: {len(all_states)}')

    