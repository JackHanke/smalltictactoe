# thanks Gemini

import csv

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

# minimax returns the tuple of the board's score, as well as (# winnining lines for 'X', # of drawing lines, # winning lines for 'O')
def minimax(board, player):
    winner = check_winner(board)
    if winner == 'X': return 1, [1,0,0]
    if winner == 'O': return -1, [0,0,1]
    if winner == 'Tie': return 0, [0,1,0]

    scores = []
    moves = []

    legal_moves = 0
    line_count = [0,0,0]
    for i in range(9):
        if board[i] == ' ':
            legal_moves += 1
            board[i] = player
            score, (num_win_lines_for_x, num_draw_lines, num_win_lines_for_o) = minimax(board, 'O' if player == 'X' else 'X')
            line_count[0] += num_win_lines_for_x
            line_count[1] += num_draw_lines
            line_count[2] += num_win_lines_for_o
            scores.append(score)
            moves.append(i)
            board[i] = ' '

    # weight line count by depth
    # line_count[0] = line_count[0]/legal_moves
    # line_count[1] = line_count[1]/legal_moves
    # line_count[2] = line_count[2]/legal_moves

    max_scores = max(scores)
    min_scores = min(scores)

    if player == 'X':
        return max_scores, line_count
    else:
        return min_scores, line_count

def get_best_move(board, player):
    best_score = -float('inf') if player == 'X' else float('inf')
    
    # Iterating 0 to 8 ensures the "Top-Most, Left-Most" tie-break rule
    best_moves = []
    tie_break = 0
    for i in range(9):
        if board[i] == ' ':
            board[i] = player
            score, (num_win_lines_for_x, num_draw_lines, num_win_lines_for_o) = minimax(board, 'O' if player == 'X' else 'X')
            # print(board)
            # print(score, (num_win_lines_for_x, num_draw_lines, num_win_lines_for_o))
            board[i] = ' '
            
            # TODO fix this horrible logic
            if player == 'X':
                if score > best_score:
                    best_score = score
                    best_moves = str(i)
                #     if score == 0:
                #         tie_break = num_win_lines_for_x
                # elif score == best_score:
                #     if score == 0:
                #         if num_win_lines_for_x > tie_break:
                #             tie_break = num_win_lines_for_x
                #             best_moves = str(i)
                #         elif num_win_lines_for_x == tie_break:
                #             best_moves += str(i)
                #     else:
                #         best_moves += str(i)

            else:
                if score < best_score:
                    best_score = score
                    best_moves = str(i)
                #     if score == 0:
                #         tie_break = num_win_lines_for_o
                # elif score == best_score:
                #     if score == 0:
                #         if num_win_lines_for_o > tie_break:
                #             tie_break = num_win_lines_for_o
                #             best_moves = str(i)
                #         elif num_win_lines_for_o == tie_break:
                #             best_moves += str(i)
                #     else:
                #         best_moves += str(i)
    
    return best_moves

def board_to_booleans(board):
    bool_board = ['0' for _ in range(18)]

    for idx, val in enumerate(board):
        if val == 'X':
            bool_board[idx] = '1'

    for idx, val in enumerate(board):
        if val == 'O':
            bool_board[idx+9] = '1'

    return bool_board

def move_to_booleans(move):
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

def create_individual_moves_dataset():
    generate_states([' '] * 9, 'X')

    for i in range(9):
        with open(f'data/datasets/{i}_moves.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'{i}' for i in range(19)])
            for state, info in sorted(all_states.items()):
                bool_board = board_to_booleans(board=state)
                if int(info[1]) == i:
                    writer.writerow(bool_board+[1])
                else:
                    writer.writerow(bool_board+[0])

# Run and Save
def create_all_dataset(path: str):
    all_states = generate_states([' '] * 9, 'X')

    
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['board_state', 'player_turn', 'best_move_index'])
        for state, info in sorted(all_states.items()):
            writer.writerow([state, info[0], "'"+info[1]+"'"])

    print(f"Dataset generated with {len(all_states)} unique board states.")

def create_espresso_dataset():
    all_states = generate_states([' '] * 9, 'X')
    
    with open("data/datasets/ttt.txt", "w") as f:
        f.write('.i 13\n')
        f.write('.o 4\n')
        for board_num, (state, (player, best_move)) in enumerate(sorted(all_states.items())):
            board_str = ''.join(board_to_booleans(board=state))
            move_str = ''.join(move_to_booleans(move=int(best_move)))
            board_num_b = format(board_num, '013b')
            move_num_b = format(int(best_move), '04b')
            # f.write(f'{board_str} {move_str}\n')
            # f.write(f'{board_num_b} {move_str}\n')
            f.write(f'{board_num_b} {move_num_b}\n')

if __name__ == '__main__':

    # create_dataset(path='data/datasets/tictactoe_best_moves_all.csv')

    # create_individual_moves_dataset()
    create_espresso_dataset()

    # get_best_move([' ']*9, 'X')
