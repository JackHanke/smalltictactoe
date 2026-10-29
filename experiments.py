import json
import random
import numpy as np
import torch
from tqdm import tqdm

from train import train_to_perfection
from data.dataset import data_json_to_tensor
from models.nn import *
from data.reps import *
from main import filter_feature_state_matrix_among_0_1
from datetime import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def intervention_experiment():
    '''
    Examination for interventions of representation, number of layers, and illegal moves masking for smallest 100% models across `num_seeds` seeds
    
    RESULTS:
        layers: 1 replen: 18 masking: False | 765
        layers: 1 replen: 18 masking: True  | 597
        layers: 1 replen: 9  masking: False | 598
        layers: 1 replen: 9  masking: True  | 541   *BEST*
        layers: 2 replen: 18 masking: False | 1205
        layers: 2 replen: 18 masking: True  | 855
        layers: 2 replen: 9  masking: False | 870
        layers: 2 replen: 9  masking: True  | 693    
    '''

    with open("data/datasets/jsons/nn_friendly_dataset.json", "r") as file:
        states_dict = json.load(file)

    num_seeds = 4
    results_dict = {}
    for num_layers in [1,2]:
        for board_rep_func, rep_length in [(binary_board_rep, 18), (trinary_board_rep, 9)]:
            for do_illegal_move_masking in [True]:
                hidden_dim = 25
                best_params = float('inf')

                perfection_reached = True
                while perfection_reached:
                    print(f'Examining hidden_dim: {hidden_dim}...')

                    for seed in range(num_seeds):
                        random.seed(seed)
                        np.random.seed(seed)       
                        torch.manual_seed(seed)

                        hidden_sizes = [hidden_dim for _ in range(num_layers)]
                        model = TicTacToeNet(
                            hidden_sizes=hidden_sizes,
                            input_size=rep_length,
                            do_illegal_move_masking=do_illegal_move_masking,
                        ).to(DEVICE)

                        dataset = tttDataset(
                            states_dict=states_dict,
                            board_rep_func=board_rep_func,
                            len_rep=rep_length,
                        )   

                        num_params = sum(p.numel() for p in model.parameters())

                        seed_perfection_reached, epoch, accuracy = train_to_perfection(
                            model=model,
                            dataset=dataset,
                            max_epochs=10_000,
                            weight_decay=0.0,
                            one_right_answer=False,
                            device=DEVICE,
                            save_checkpoint=False,
                            verbose=True,
                        )

                        if not seed_perfection_reached: 
                            perfection_reached = False
                            break
                        elif seed == num_seeds - 1:
                            perfection_reached = True

                    if perfection_reached:
                        best_params = num_params
                        hidden_dim -= 1

                checkpoint_path = f'models/checkpoints/intervention_experiment/nn_{best_params}_{hidden_sizes}_{num_layers}_{rep_length}_{do_illegal_move_masking}.pth'
                model.zero_grad() # zero grads for file size
                torch.save(model.state_dict(), checkpoint_path)
                # print(f'Model saved at: {checkpoint_path}')

                results_dict[f'{num_layers}_{rep_length}_{do_illegal_move_masking}'] = best_params
                print(f'> layers: {num_layers} replen: {rep_length} masking: {do_illegal_move_masking} | {best_params}')


def smallest_possible_experiment():
    '''
    Find best seed with best settings over `num_seeds`==30, single layer trinary rep with illegal move masking

    hidden_dim=24 seems to be the limit, 100 seeds tried with 23
    '''

    DATA_PATH = "data/datasets/jsons/other.json"
    print(f'DATA PATH: {DATA_PATH}')
    
    board_rep_fn, rep_length = (trinary_board_rep, 9)
    data_tensor, moves_mask = data_json_to_tensor(
        data_json_path=DATA_PATH,
        board_rep_fn=board_rep_fn,
    )

    hidden_dim = 9
    print(f'Starting at hidden_dim: {hidden_dim}')
    num_layers = 1
    best_params = float('inf')

    num_seeds = 100 # 

    seed_found = True
    while seed_found:
        seed_found = False
        for seed in range(num_seeds):
        # for seed in range(num_seeds):
            random.seed(seed)
            np.random.seed(seed)       
            torch.manual_seed(seed)

            # board_rep_func, rep_length = (trinary_plus_sym_board_rep, 17)
            do_illegal_move_masking = True

            hidden_sizes = [hidden_dim for _ in range(num_layers)]
            model = TicTacToeNet(
                hidden_sizes=hidden_sizes,
                input_size=rep_length,
                do_illegal_move_masking=do_illegal_move_masking,
            ).to(DEVICE)

            num_params = sum(p.numel() for p in model.parameters())


            seed_perfection_reached, epoch, accuracy = train_to_perfection(
                model=model,
                data_tensor=data_tensor,
                moves_mask=moves_mask,
                max_epochs=25_000,
                weight_decay=0.0,
                one_right_answer=False,
                device=DEVICE,
                save_checkpoint=False,
                verbose=True,
                seed = seed,
            )

            if seed_perfection_reached: 
                print(f'Param {num_params} found seed: {seed}')
                seed_found = True
                checkpoint_path = f'models/checkpoints/smallest_possible_experiment/nn_{num_params}_{hidden_sizes}_{num_layers}_{rep_length}_{do_illegal_move_masking}_{seed}.pth'
                model.zero_grad() # zero grads for file size
                torch.save(model.state_dict(), checkpoint_path)
                break
            elif seed == num_seeds - 1:
                seed_found = False

        if seed_found:
            best_params = num_params
            hidden_dim -= 1
            print(f'Seed Found! Testing hidden_dim: {hidden_dim}')

    print(f'Best params from seed hunt: {best_params}')

def further_training_perfection_experiment():
    '''
    
    '''

    DATA_PATH = "data/datasets/jsons/other.json"
    print(f'DATA PATH: {DATA_PATH}')
    
    board_rep_fn, rep_length = (trinary_board_rep, 9)
    data_tensor, moves_mask = data_json_to_tensor(
        data_json_path=DATA_PATH,
        board_rep_fn=board_rep_fn,
    )

    model = TicTacToeNet(
        hidden_sizes=[9],
        input_size=9
    ).to(DEVICE)
    PATH = 'models/checkpoints/nnother_[9, 9, 9]_180_2026-10-25-21:10:15.pth'
    # PATH = 'models/checkpoints/pruning/nn_104_[5]_1_9.pth'
    model.load_state_dict(torch.load(PATH, weights_only=True))

    train_to_perfection(
        model=model,
        data_tensor=data_tensor,
        moves_mask=moves_mask,
        max_epochs=110_000,
        weight_decay=0.0,
        one_right_answer=False,
        device=DEVICE,
        save_checkpoint=True,
        verbose=True,
        to_perfection=False, # NOTE this is what changes here
        name='other_extend',
    )

def simple_train_loop(
        model,
        state_tensor,
        y,
        max_epochs: int = 10_000,
        save_checkpoint: bool = True,
        name: str = '',
        seed: int = None,
    ):
    checkpoint_time = datetime.now()
    checkpoint_time_str = checkpoint_time.strftime("%Y-%m-%d-%H:%M:%S")

    num_params = sum(p.numel() for p in model.parameters())
    n = state_tensor.shape[0] # number of trianing points
    
    criterion = nn.CrossEntropyLoss()
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 0.0
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.9,
        patience=500,
        min_lr=1e-5,
    )

    prog = tqdm(range(max_epochs), total=max_epochs)
    for epoch in prog:
        outputs = model(state_tensor)
        # print(outputs.shape)
        # print(y_0.shape)
        #
        loss = criterion(outputs, y)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        predictions = torch.argmax(outputs, dim=1)

        correct = (predictions == y).sum()

        accuracy = 100 * correct / n
        
        if epoch % 100 == 0 or accuracy == 100.0:

            prog_str = f'Loss: {loss.item():.8f}, Accuracy: {accuracy:.4f}%, Correct: {correct}, {n - correct}/{n} remaining, n_params: {num_params}.'
            if seed is not None: prog_str += f' Seed: {seed}'
            prog.set_description(prog_str)

            if accuracy == 100.0:
                perfection_reached = True
                if save_checkpoint:
                    checkpoint_path = f'models/checkpoints/nn_{name}_{model.hidden_sizes}_{num_params}_{checkpoint_time_str}.pth'
                    model.zero_grad() # zero grads for file size
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f'Model saved at: {checkpoint_path}')
                    print(f'Loss: {loss.item():.5f}')

                return perfection_reached, epoch, accuracy

    return False, epoch, accuracy

def corner_and_edge_classifier_creator():
    DATA_PATH = 'data/datasets/jsons/other.json'
    with open(DATA_PATH, 'r') as f: states = json.load(f)

    board_rep_fn, rep_length = (trinary_board_rep, 9)
    data_tensor, moves_mask = data_json_to_tensor(
        data_json_path=DATA_PATH,
        board_rep_fn=board_rep_fn,
    )

    move = 0
    # move = 1
    print(f'Move: {move}')

    num_seeds = 20
    num_moves_to_resolve = 15
    # prog = tqdm(itertools.product([0, 1], repeat=num_moves_to_resolve), total=2**num_moves_to_resolve)
    prog = [[1 for _ in range(100)]]
    for inclusion_code in prog:
        # generate datasets, resolving positions with both 0 and 1 optimal 
        state_matrix_0, y_0, state_matrix_1, y_1 = filter_feature_state_matrix_among_0_1(states, data_tensor.numpy(), inclusion_code)

        state_matrix_0 = torch.from_numpy(state_matrix_0).to(DEVICE, non_blocking=True)
        state_matrix_0 = state_matrix_0[:, [1,2,3,4,5,6,7,8]]
        y_0 = torch.tensor(y_0).to(DEVICE, non_blocking=True)

        state_matrix_1 = torch.from_numpy(state_matrix_1).to(DEVICE, non_blocking=True)
        state_matrix_1 = state_matrix_1[:, [0,2,3,4,5,6,7,8]]
        y_1 = torch.tensor(y_1).to(DEVICE, non_blocking=True)

        for seed in range(num_seeds):
            # for seed in range(num_seeds):
            random.seed(seed)
            np.random.seed(seed)       
            torch.manual_seed(seed)

            if move == 0:

                corner_model = tttclassifer(
                    input_size=8,
                    hidden_sizes=[7],
                ).to(DEVICE)
                
                simple_train_loop(
                    model=corner_model,
                    state_tensor=state_matrix_0,
                    y=y_0,
                    max_epochs=50_000,
                    name='cornerclass',
                    seed=seed,
                )

            elif move == 1:
                edge_model = tttclassifer(
                    input_size=8,
                    hidden_sizes=[5],
                ).to(DEVICE)

                simple_train_loop(
                    model=edge_model,
                    state_tensor=state_matrix_1,
                    y=y_1,
                    max_epochs=30_000,
                    name='edgeclass',
                    seed=seed,
                )
        
        

if __name__ == '__main__':

    # intervention_experiment()

    # smallest_possible_experiment()

    # further_training_perfection_experiment()

    corner_and_edge_classifier_creator()