# thanks Gemini kinda
import csv
import pandas as pd
from game import generate_states_from_root_board
from reps import binary_board_rep

def create_dataset_df(
        seed: int = None,
        return_all_bests=True,
    ):
    ''' create dataset using pandas '''
    all_states = generate_states_from_root_board(
        [' '] * 9,
        'X',
        seed=seed,
        return_all_bests=return_all_bests,
    )

    # data = []
    # for state, moves in sorted(all_states.items()):
    #     board_rep = binary_board_rep(state)
    #     row = {}
    #     for idx, board_dig in enumerate(board_rep):
    #         row[str(idx)] = board_dig
    #     row['best_move_index'] = int(moves)
    #     data.append(row)
    
    df = pd.DataFrame(data)
    return df


# def create_dataset(path: str, seed: int = None):
#     generate_states([' '] * 9, 'X', seed=seed)

#     with open(path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([f'{i}' for i in range(18+1)])
#         for state, (player, best_move) in sorted(all_states.items()):
#             ## board representation
#             board_rep = bin_rep(board=state)
#             # board_rep = nn_rep(board=state)

#             ## move representation
#             # move_rep = one_hot_move(move=int(best_move))
#             move_rep = [int(best_move)]

#             ## write
#             writer.writerow(board_rep+move_rep)


# def create_all_dataset(path: str):
#     ''' create dataset  '''
#     all_states = generate_states([' '] * 9, 'X')
    
#     with open(path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['board_state', 'player_turn', 'best_move_index'])
#         for state, info in sorted(all_states.items()):

#             writer.writerow([state, info[0], "'"+info[1]+"'"])

#     print(f"Dataset generated with {len(all_states)} unique board states.")

# def create_espresso_dataset(path: str):
#     ''' dataset for binary circuit minimization with espresso '''
#     all_states = generate_states([' '] * 9, 'X')
    
#     with open(path, "w") as f:
#         f.write('.i 13\n')
#         f.write('.o 4\n')
#         for board_num, (state, (player, best_move)) in enumerate(sorted(all_states.items())):
#             board_str = ''.join(bin_rep(board=state))
#             move_str = ''.join(one_hot_move(move=int(best_move)))
#             board_num_b = format(board_num, '013b')
#             move_num_b = format(int(best_move), '04b')
#             # f.write(f'{board_str} {move_str}\n')
#             # f.write(f'{board_num_b} {move_str}\n')
#             f.write(f'{board_num_b} {move_num_b}\n')

if __name__ == '__main__':

    seed = 0
    df = create_dataset_df(
        seed=seed,
        return_all_bests=True,
    )
    print(df)
    # df.to_csv(f'data/datasets/ttt_best_bin_{seed}.csv')


    # create_dataset(path=PATH)
    # create_all_dataset(path=PATH)
    # create_espresso_dataset()