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

def minimax(board, player):
    winner = check_winner(board)
    if winner == 'X': return 1
    if winner == 'O': return -1
    if winner == 'Tie': return 0

    scores = []
    moves = []
    
    for i in range(9):
        if board[i] == ' ':
            board[i] = player
            score = minimax(board, 'O' if player == 'X' else 'X')
            scores.append(score)
            moves.append(i)
            board[i] = ' '

    if player == 'X':
        return max(scores)
    else:
        return min(scores)

def get_best_move(board, player):
    best_score = -float('inf') if player == 'X' else float('inf')
    best_move = -1
    
    # Iterating 0 to 8 ensures the "Top-Most, Left-Most" tie-break rule
    for i in range(9):
        if board[i] == ' ':
            board[i] = player
            score = minimax(board, 'O' if player == 'X' else 'X')
            board[i] = ' '
            
            if player == 'X':
                if score > best_score:
                    best_score = score
                    best_move = i
            else:
                if score < best_score:
                    best_score = score
                    best_move = i
    return best_move

# Generate all legal reachable states
all_states = {}

def generate_states(board, player):
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

# Run and Save
generate_states([' '] * 9, 'X')

with open('tictactoe_best_moves.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['board_state', 'player_turn', 'best_move_index'])
    for state, info in sorted(all_states.items()):
        writer.writerow([state, info[0], info[1]])

print(f"Dataset generated with {len(all_states)} unique board states.")