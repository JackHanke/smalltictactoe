## 

import json
import itertools
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from scipy.optimize import linprog
from scipy.special import binom

from data.reps import *


def valid_subsets(list_of_lists, max_threshold):
    # Pre-extract lengths to avoid repeating len() during recursion
    lengths = [len(sub) for sub in list_of_lists]
    n = len(list_of_lists)
    
    def backtrack(index, current_subset, current_length):
        # Yield the current valid subset
        yield list(current_subset)
        
        for i in range(index, n):
            sub_len = lengths[i]
            # PRUNING STEP: Only recurse if adding this sublist stays <= max_threshold
            if current_length + sub_len <= max_threshold:
                current_subset.append(list_of_lists[i])
                yield from backtrack(i + 1, current_subset, current_length + sub_len)
                current_subset.pop()  # Backtrack

    yield from backtrack(0, [], 0)

def get_syms(r):
    # using notation from "F" diagram at: https://en.wikipedia.org/wiki/Dihedral_group_of_order_8
    syms = [
        (# e
            (r[0][0],r[0][1], r[0][2]),
            (r[1][0],r[1][1], r[1][2]),
            (r[2][0],r[2][1], r[2][2]),
        ),
        ( # a
            (r[2][0],r[1][0], r[0][0]),
            (r[2][1],r[1][1], r[0][1]),
            (r[2][2],r[1][2], r[0][2]),
        ),
        ( # a2
            (r[2][2],r[2][1], r[2][0]),
            (r[1][2],r[1][1], r[1][0]),
            (r[0][2],r[0][1], r[0][0]),
        ),
        ( # a3
            (r[0][2],r[1][2], r[2][2]),
            (r[0][1],r[1][1], r[2][1]),
            (r[0][0],r[1][0], r[2][0]),
        ),
        ( # b
            (r[0][2],r[0][1], r[0][0]),
            (r[1][2],r[1][1], r[1][0]),
            (r[2][2],r[2][1], r[2][0]),
        ),
        ( # ab
            (r[0][0],r[1][0], r[2][0]),
            (r[0][1],r[1][1], r[2][1]),
            (r[0][2],r[1][2], r[2][2]),
        ),
        ( # a2b
            (r[2][0],r[2][1], r[2][2]),
            (r[1][0],r[1][1], r[1][2]),
            (r[0][0],r[0][1], r[0][2]),
        ),
        ( # a3b
            (r[2][2],r[1][2], r[0][2]),
            (r[2][1],r[1][1], r[0][1]),
            (r[2][0],r[1][0], r[0][0]),
        ),
    ]

    return set(syms)

def is_linearly_separable(X, y):
    """
    Checks if a dataset is linearly separable using Linear Programming.
    X: 2D array-like of shape (n_samples, n_features)
    y: 1D array-like of shape (n_samples,) with two distinct class values
    Returns: True if separable, False otherwise.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Map labels to {-1, +1}
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("Data must contain exactly 2 classes.")
    y_binary = np.where(y == classes[0], -1, 1)
    
    n_samples, n_features = X.shape
    
    # Construct inequality constraints: -y_i * (w^T * x_i + b) <= -1
    # Constraint matrix A_ub aligns with [w_0, w_1, ..., w_d, b]
    A_ub = np.hstack([-y_binary[:, None] * X, -y_binary[:, None]])
    b_ub = -np.ones(n_samples)
    
    # Dummy objective (we only care about feasibility)
    c = np.zeros(n_features + 1)
    
    # Solve LP (bounds=None allows w and b to take negative values)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method='highs')
    
    return res.success, res.x


def evaluate_monomial(board, powers):
    """Evaluates a single monomial (represented by a tuple of powers) on a board."""
    val = 1
    for b_val, p in zip(board, powers):
        if p > 0: val *= (b_val ** p)
    return val


def monomial_to_feature_str(monomial):
    ''' get name of monomial in string form '''
    codes = ['c1','e1','c2','e2','c','e3','c3','e4','c4']

    feature_str = ''
    for expon, code in zip(monomial, codes):
        if expon == 1:
            feature_str += code
        elif expon == 2:
            feature_str += f'({code}^2)'
    
    return feature_str
            

def build_feature_state_matrix(states, max_degree: int = 3):
    '''
    build feature x state matrix, as well as monomials and grouped monomials (by symmetry)
    '''
    boards = []
    for board, moves in states.items():
        board_arr = []
        for charac in board:
            if charac == ' ':
                board_arr.append(0)
            elif charac == 'X':
                board_arr.append(1)
            elif charac == 'O':
                board_arr.append(-1)
        boards.append(board_arr)

    # valid_powers = [i for i in range(max_degree)]
    valid_powers = [0 ,1]
    all_monomials = [
        p for p in itertools.product(valid_powers, repeat=9)
        if 0 < sum(p) <= max_degree
    ]
    all_monomial_groups = []

    feature_state_matrix = np.zeros((len(boards), len(all_monomials)), dtype=int)
    for j, powers in enumerate(all_monomials):
        # build feature state matrix
        for i, board in enumerate(boards):
            feature_state_matrix[i, j] = evaluate_monomial(board, powers)

        # populate all_monomials_groups
        reformatted_powers = [list(powers[3*i:3*i+3]) for i in range(3)]
        syms_set = get_syms(reformatted_powers)
        names = []
        for elem in syms_set:
            name = []
            for i in elem:
                name.extend(list(i))
            names.append(monomial_to_feature_str(name))
        if set(names) not in all_monomial_groups:
            all_monomial_groups.append(set(names))

    all_monomial_groups = [list(name_set) for name_set in all_monomial_groups]

    return feature_state_matrix, all_monomials, all_monomial_groups

    
def filter_feature_state_matrix(states, feature_state_matrix, move_of_interest):
    '''
    
    '''
    y = []
    rows_to_keep = []
    for idx, (board, moves) in enumerate(states.items()):
        # if more than one legal move, does not include the center (4), and the move of interest is optimal
        if (board.count(' ') > 1) and (4 not in moves) and move_of_interest in moves:
            rows_to_keep.append(idx)
            y.append(1)
        # else if this move is legal, there is more than one legal move, and the move of interest is not optimal
        elif board[move_of_interest] == ' ' and (board.count(' ') > 1) and move_of_interest not in moves:
            rows_to_keep.append(idx)
            y.append(-1)

    return feature_state_matrix[rows_to_keep], y

def filter_feature_state_matrix_among_0_1(states, feature_state_matrix, inclusion_code):
    '''
    filters the full feature x state matrix for creating datasets for the move 0 and move 1 classifiers, accepting the inclusion code for which ambiguous move should be included
    '''
    count = 0
    y_s = [[],[]]
    rows_to_keep_s = [[],[]]
    for idx, (board, moves) in enumerate(states.items()):
        # if more than one legal move, does not include the center (4), ...
        # ... and both 0 and 1 are optimal
        if (board.count(' ') > 1) and (4 not in moves) and (0 in moves and 1 in moves) and len(moves) != board.count(' '):
            resolved_move = inclusion_code[count]
            count += 1
            rows_to_keep_s[resolved_move].append(idx)
            y_s[resolved_move].append(1)

        # ... only 0 is optimal
        elif (board.count(' ') > 1) and (4 not in moves) and (0 in moves) and len(moves) != board.count(' '):
            rows_to_keep_s[0].append(idx)
            y_s[0].append(1)

        # ... only 1 is optimal
        elif (board.count(' ') > 1) and (4 not in moves) and (1 in moves) and len(moves) != board.count(' '):
            rows_to_keep_s[1].append(idx)
            y_s[1].append(1)

        # if 0 is legal, there is more than one legal move, and 0 is not optimal
        if board[0] == ' ' and (board.count(' ') > 1) and 0 not in moves:
            rows_to_keep_s[0].append(idx)
            y_s[0].append(-1)

        # else if 1 is legal, there is more than one legal move, and 0 is not optimal
        elif board[1] == ' ' and (board.count(' ') > 1) and 1 not in moves:
            rows_to_keep_s[1].append(idx)
            y_s[1].append(-1)

    # 
    return feature_state_matrix[rows_to_keep_s[0]], y_s[0], feature_state_matrix[rows_to_keep_s[1]], y_s[1]


def islinsep_recurse(
        state_matrices,
        y_s,
        column_indices,
        sols,
        memoize,
        verbose: bool = False,
    ):
    '''
    top down remove-one-at-a-time for state matrices
    '''
    state_tuple = tuple(column_indices)
    if state_tuple in memoize: return
    memoize.add(state_tuple)

    res_x = []
    is_lin_sep = True
    for state_matrix, y in zip(state_matrices, y_s):
        is_lin_sep_ind, res_x_ind = is_linearly_separable(state_matrix[:, column_indices], y)
        res_x.append(res_x_ind)
        is_lin_sep = is_lin_sep and is_lin_sep_ind


    if is_lin_sep:
        first_sol_indices = sols[0][0]
        # 
        if len(column_indices) < len(first_sol_indices): 
            sols[:] = [(column_indices, res_x)]
            if verbose: print(f'{len(column_indices)} {column_indices} {res_x}')
        elif len(column_indices) == len(first_sol_indices):
            sols.append((column_indices, res_x))
            if verbose: print(f'{len(column_indices)} {column_indices} {res_x}')
    else:
        return

    # try to remove one of the columns
    for i in range(len(column_indices)):
        reduced_column_indices = deepcopy(column_indices)
        reduced_column_indices.pop(i)

        islinsep_recurse(
            state_matrices,
            y_s,
            reduced_column_indices,
            sols,
            memoize=memoize
        )

def islinsep_k_subsets(
        state_matrices,
        y_s,
        column_groups,
        k,
        early_stop: bool = False,
    ): 
        '''
        
        '''
        sols = []
        
        prog = tqdm(itertools.combinations(column_groups, k), total=binom(len(column_groups), k))
        prog.set_description(f'k={k} {len(sols)} solutions found.')
        for column_groups_subset in prog:
            res_x = []
            is_lin_sep = True
            for state_matrix, y in zip(state_matrices, y_s):
                is_lin_sep_ind, res_x_ind = is_linearly_separable(state_matrix[:, column_groups_subset], y)
                res_x.append(res_x_ind)
                is_lin_sep = is_lin_sep and is_lin_sep_ind

            if is_lin_sep:
                sols.append((column_groups_subset, res_x))
                prog.set_description(f'k={k}, {len(sols)} solutions found.')
                if early_stop: return sols
                
        return sols


def islinsep_smallest_groups_first(
        mon_str_dict,
        state_matrix,
        y,
        column_groups,
        max_threshold: int = 26,
        early_stop: bool = False,
        verbose: bool = False,
    ):
        '''
        bottom up linearity search, where the 
        '''
        sols = []
        current_best = float('inf')

        for column_groups_subset in valid_subsets(column_groups, max_threshold=max_threshold):
            column_indices_subset = [mon_str_dict[name] for group in column_groups_subset for name in group]
            is_lin_sep, res_x = is_linearly_separable(state_matrix[:, column_indices_subset], y)

            if is_lin_sep:
                if len(column_indices_subset) < current_best:
                    current_best = len(column_indices_subset)
                    sols = [(column_groups_subset, res_x, res_x)]
                    if verbose: print(f'### new best: {current_best} ###\n{(column_groups_subset, res_x, res_x)}')
                    if early_stop: return sols
                elif len(column_indices_subset) == current_best:
                    sols.append((column_groups_subset, res_x, res_x))
                
        return sols


def feature_string_to_string_parts(feature_str):
    if len(feature_str) == 2:
        string_parts = [feature_str]
    elif len(feature_str) == 3:
        if feature_str[2] == 'c':
            string_parts = [feature_str[0:2], feature_str[2]]
        elif feature_str[0] == 'c' and feature_str[1] in ('c','e'):
            string_parts = [feature_str[0], feature_str[1:3]]
    elif len(feature_str) == 4:
        string_parts = [feature_str[0:2], feature_str[2:4]]
    else:
        raise Exception('Uh oh!')
    
    return string_parts

def rotate_feature_clockwise(feature_str: str, mon_str_dict):
    # NOTE awful, but only way i could think to do it

    string_parts = feature_string_to_string_parts(feature_str)

    rotate_map = {
        'c':'c',
        'c1':'c2',
        'c2':'c4',
        'c3':'c1',
        'c4':'c3',
        'e1':'e3',
        'e2':'e1',
        'e3':'e4',
        'e4':'e2',
    }

    rotated_string_parts = []
    for part in string_parts:
        rotated_string_parts.append(rotate_map[part])

    if len(rotated_string_parts) > 1:
        # check if this or the swap is in the conversion dict
        temp_rotated_feature_str = "".join(rotated_string_parts)
        temp_flipped_rotated_feature_str = "".join([rotated_string_parts[1], rotated_string_parts[0]])
        if temp_rotated_feature_str in mon_str_dict:
            rotated_feature_str = temp_rotated_feature_str
        elif temp_flipped_rotated_feature_str in mon_str_dict:
            rotated_feature_str = temp_flipped_rotated_feature_str
        else:
            raise Exception('This shouldnt happen')
    else:
        rotated_feature_str = "".join(rotated_string_parts)
    return rotated_feature_str


def make_matix():

    # create single feature state matrix
    feature_state_matrix, all_monomials, all_monomial_groups = build_feature_state_matrix(states, max_degree=2)

    # create feature indexing map
    mon_str_dict = {monomial_to_feature_str(mon):idx for idx, mon in enumerate(all_monomials)} | {idx:monomial_to_feature_str(mon) for idx, mon in enumerate(all_monomials)}

    # define what groups to start the search with

    starter_groups = [
        ['c1c4', 'c2c3'],
        ['c1', 'c2', 'c4', 'c3'],
        ['c1c', 'c2c', 'cc4', 'cc3'],
        ['e1c', 'ce3', 'ce4', 'e2c'],
        ['e1e2', 'e1e3', 'e3e4', 'e2e4'],
        ['c1e1', 'c2e3', 'e4c4', 'e2c3', 'c1e2', 'c3e4', 'e3c4', 'e1c2'],
    ]
    starter_indices = [mon_str_dict[name] for group in starter_groups for name in group]
    print(f'starter indices: {starter_indices}')

    zero_classifier_idxs = [0, 21, 3, 35, 25, 2, 18, 8, 17, 11, 13, 14, 33, 20, 12, 24]
    zero_classifier_coefs = [40,  30,  34,  70, 6,  -2,  80,  66,  -2, 110,  -8, 26, -22, -38,  24, -32, 5]

    one_classifier_idxs =  [36, 0, 21, 5, 7, 17, 14, 20]
    one_classifier_coefs =  [18, 6,  20, -16,  24, -16, -54, -70, -19]

    mat = np.zeros((8, len(starter_indices)+1), dtype=np.int32)



    # one classifier coefs
    for idx, coef in zip(zero_classifier_idxs, zero_classifier_coefs[:-1]):

        # convert global monomial index to matrix column index
        feature_name = mon_str_dict[idx]

        for i in range(0,4):
            # print((i, feature_name))

            temp = mon_str_dict[feature_name]
            mat_idx = starter_indices.index(temp)
            mat[i][mat_idx] = coef

            # need to rotate mat_idx
            feature_name = rotate_feature_clockwise(feature_name, mon_str_dict)

    # one classifier coefs
    for idx, coef in zip(one_classifier_idxs, one_classifier_coefs[:-1]):

        # convert global monomial index to matrix column index
        feature_name = mon_str_dict[idx]

        for i in range(4,8):
            # print((i, feature_name))

            temp = mon_str_dict[feature_name]
            mat_idx = starter_indices.index(temp)
            mat[i][mat_idx] = coef

            # need to rotate mat_idx
            feature_name = rotate_feature_clockwise(feature_name, mon_str_dict)

    # biases for 0 and 1
    for i in range(0,4):
        mat[i][-1] = zero_classifier_coefs[-1]
    for i in range(4,8):
        mat[i][-1] = one_classifier_coefs[-1]

    return mat


def board_str_to_feature_vec(board_str, add_bias_one: bool = False):

    # def board_to_feature_vector(board_str):
    rep = trinary_board_rep(board_str)

    starter_groups = [
        ['c1c4', 'c2c3'],
        ['c1', 'c2', 'c4', 'c3'],
        ['c1c', 'c2c', 'cc4', 'cc3'],
        ['e1c', 'ce3', 'ce4', 'e2c'],
        ['e1e2', 'e1e3', 'e3e4', 'e2e4'],
        ['c1e1', 'c2e3', 'e4c4', 'e2c3', 'c1e2', 'c3e4', 'e3c4', 'e1c2'],
    ]
    flattened_starter_groups = [name for group in starter_groups for name in group]

    feature_idx_map = {
        'c1':0,
        'e1':1,
        'c2':2,
        'e2':3,
        'c':4,
        'e3':5,
        'c3':6,
        'e4':7,
        'c4':8,
    }
    feature_vec = []
    for feature_str in flattened_starter_groups:
        string_parts = feature_string_to_string_parts(feature_str)

        prod = 1
        for term in string_parts:
            prod *= rep[feature_idx_map[term]]
        feature_vec.append(prod)

    if add_bias_one: feature_vec.append(1)

    return np.array(feature_vec, dtype=np.int32)


def decision_function(board_str):
    inf_mask_val = -9999

    mat = [[  0, -32,   0,  30,  40,  34,   0,   0, 110,  -8, -22,  26,  24, -38,   0,   0,  66,  -2,   0,   6,  -2,  80,   0,   0,   0,  70,   5],
        [-32,   0,  34,   0,  30,  40,  -8,   0,   0, 110, -38, -22,  26,  24,  -2,   0,   0,  66,  80,   0,   6,  -2,   0,   0,  70,   0,   5],
        [  0, -32,  40,  34,   0,  30, 110,  -8,   0,   0,  24, -38, -22,  26,  66,  -2,   0,   0,  -2,  80,   0,   6,   0,  70,   0,   0,   5],
        [-32,   0,  30,  40,  34,   0,   0, 110,  -8,   0,  26,  24, -38, -22,   0,  66,  -2,   0,   6,  -2,  80,   0,  70,   0,   0,   0,   5],
        [  0,   0,  18,  20,   6,   0,   0,   0,   0,   0,   0, -54,   0, -70,   0,   0,   0, -16,   0,   0,   0,   0,   0, -16,  24,   0, -19],
        [  0,   0,   0,  18,  20,   6,   0,   0,   0,   0, -70,   0, -54,   0, -16,   0,   0,   0,   0,   0,   0,   0, -16,  24,   0,   0, -19],
        [  0,   0,   6,   0,  18,  20,   0,   0,   0,   0,   0, -70,   0, -54,   0, -16,   0,   0,   0,   0,   0,   0,  24,   0,   0, -16, -19],
        [  0,   0,  20,   6,   0,  18,   0,   0,   0,   0, -54,   0, -70,   0,   0,   0, -16,   0,   0,   0,   0,   0,   0,   0, -16,  24, -19]]
    mat = np.array(mat)

    feature_vec = board_str_to_feature_vec(board_str, add_bias_one=True)

    matprod = mat @ feature_vec.T

    matrix_move_code = [0,2,8,6,1,5,7,3]

    ## mask illegal moves
    illegal_idxs = [i for i, c in enumerate(board_str) if c != ' ']
    is_illegal = []
    for code in matrix_move_code:
        if code in illegal_idxs: is_illegal.append(True)
        else: is_illegal.append(False)
    is_illegal = np.array(is_illegal, dtype=np.bool)
    matprod[is_illegal] = inf_mask_val
    
    ## process vector
    moves_produced = []
    for idx, val in zip(matrix_move_code, matprod.tolist()):
        # if greater than 0 element, this move is optimal
        if val > 0:
            moves_produced.append(idx)

    # 
    if len(moves_produced) == 0:
        # if the center is illegal 
        if 4 in illegal_idxs:
            for idx, val in zip(matrix_move_code, matprod.tolist()):
                # if less than 0 element, this move is optimal
                if val < 0 and val > inf_mask_val:
                    moves_produced.append(idx)
            # print('umm')
        # else move to the center
        else:
            moves_produced.append(4)

    return moves_produced


def main():
    '''
    main function for finding a small human-memorizable solution to non-trivial tictactoe positions
    '''

    # create single feature state matrix
    feature_state_matrix, all_monomials, all_monomial_groups = build_feature_state_matrix(states, max_degree=2)

    # create feature indexing map
    mon_str_dict = {monomial_to_feature_str(mon):idx for idx, mon in enumerate(all_monomials)} | {idx:monomial_to_feature_str(mon) for idx, mon in enumerate(all_monomials)}

    # define what groups to start the search with
    starter_groups = all_monomial_groups
    starter_indices = [mon_str_dict[name] for group in starter_groups for name in group]

    # for all possible ways to resolve 0 1 optimality
    num_moves_to_resolve = 3
    max_threshhold = 30
    best_so_far = float('inf')
    prog = tqdm(itertools.product([0, 1], repeat=num_moves_to_resolve), total=2**num_moves_to_resolve)
    for inclusion_code in prog:

        # generate datasets, resolving positions with both 0 and 1 optimal 
        state_matrix_0, y_0, state_matrix_1, y_1 = filter_feature_state_matrix_among_0_1(states, feature_state_matrix, inclusion_code)

        is_lin_sep_0, res_x_0 = is_linearly_separable(state_matrix_0[:, starter_indices], y_0)
        is_lin_sep_1, res_x_1 = is_linearly_separable(state_matrix_1[:, starter_indices], y_1)
        
        memoize = set()
        sols = [(all_monomial_groups, res_x_0, res_x_1)]

        # islinsep_recurse(
        #     mon_str_dict,
        #     [state_matrix_0, state_matrix_1],
        #     [y_0, y_1],
        #     column_groups=all_monomial_groups,
        #     sols_found=sols,
        #     memoize=memoize
        # )


        # find first combination of features 
        # sols = is_linearly_separable_smallest_groups_first_both(
        #     mon_str_dict,
        #     state_matrix_0,
        #     state_matrix_1,
        #     y_0,
        #     y_1,
        #     starter_groups,
        #     max_threshold = max_threshhold,
        #     early_stop = True,
        # )

        # if len(sols) > 0:
        #     features_found = sols[0][0]
        #     num_features_found = sum([len(subgroup) for subgroup in features_found])
            # if num_features_found <= best_so_far:
            #     best_so_far = num_features_found
            #     prog.set_description(f'Best so far: {best_so_far}')
            # print(f'### {num_features_found} ###')
            # print(features_found)
            # print(inclusion_code)

            # print(f'Inclusion code {inclusion_code} has feature count: {num_features_found}')

    prog = tqdm(itertools.product([0, 1], repeat=num_moves_to_resolve), total=2**num_moves_to_resolve)
    best_so_far = float('inf')
    inclusion_code = (1,1,1)
    starter_groups = [['c1', 'c4', 'c2', 'c3'], ['c3e4', 'c1e1', 'e3c4', 'e1c2', 'c2e3', 'e4c4', 'c1e2', 'e2c3'], ['e1e2', 'e1e3', 'e3e4', 'e2e4'], ['c1c', 'cc4', 'cc3', 'c2c'], ['ce3', 'e1c', 'e2c', 'ce4'], ['c1c4', 'c2c3']]
    # starter_groups = [['c4', 'c2', 'c1', 'c3'], ['c2e3', 'e1c2', 'c1e2', 'e3c4', 'c3e4', 'e2c3', 'c1e1', 'e4c4'], ['c1c2', 'c2c4', 'c3c4', 'c1c3'], ['e1e3', 'e3e4', 'e1e2', 'e2e4'], ['c1c', 'cc4', 'cc3', 'c2c'], ['e1c', 'e2c', 'ce4', 'ce3'], ['e2e3', 'e1e4'], ['c1c4', 'c2c3']]
    # (0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    state_matrix_0, y_0, state_matrix_1, y_1 = filter_feature_state_matrix_among_0_1(states, feature_state_matrix, inclusion_code)
    starter_indices = [mon_str_dict[name] for group in starter_groups for name in group]

    # further reduce separetly
    # sols = is_linearly_separable_smallest_groups_first(
    #     mon_str_dict,
    #     state_matrix_0,
    #     y_0,
    #     starter_groups,
    #     max_threshold = max_threshhold,
    #     early_stop = True,
    # )


    k = 10
    sols = []
    while len(sols) == 0:
        k += 1
        sols = islinsep_k_subsets(
            [state_matrix_0],
            [y_0],
            starter_indices,
            k,
            early_stop=True
        )

        '''
        c1 e1 c2
        e2 c  e3
        c3 e4 c4
        
        [
            ['c1', 'c4', 'c2', 'c3'],
            ['c3e4', 'c1e1', 'e3c4', 'e1c2', 'c2e3', 'e4c4', 'c1e2', 'e2c3'],
            ['e1e2', 'e1e3', 'e3e4', 'e2e4'],
            ['c1c', 'cc4', 'cc3', 'c2c'],
            ['ce3', 'e1c', 'e2c', 'ce4'],
            ['c1c4', 'c2c3']
        ]

        [
            36, 21, 0, 3,
            5, 44, 7, 35, 25, 2, 42, 18,
            34, 32, 8, 17,
            41, 11, 13, 26,
            14, 33, 20, 12,
            37, 24
        ]

        [
            ['c1', 'c2', 'c4', 'c3'],
            ['c3e4', 'c1e1', 'e3c4', 'e1c2', 'c2e3', 'e4c4', 'c1e2', 'e2c3'],
            ['e1e2', 'e1e3', 'e3e4', 'e2e4'],
            ['c1c', 'c2c', 'cc4', 'cc3'],
            ['e1c', 'ce3', 'ce4', 'e2c'],
            ['c1c4', 'c2c3']
        ]

        [
            36, 0, 21, 3,
            5, 44, 7, 35, 25, 2, 42, 18,
            34, 32, 8, 17,
            41, 26, 11, 13, 
            33, 14, 12, 20,
            37, 24
        ]
        
        0 classifier simplified (k=16, 60% of the way through)
        ((0, 21, 3, 35, 25, 2, 18, 8, 17, 11, 13, 14, 33, 20, 12, 24),
        array([ 40.,  30.,  34.,  70.,   6.,  -2.,  80.,  66.,  -2., 110.,  -8., 26., -22., -38.,  24., -32.,   5.])


        1 classifier simplified
        [((36, 0, 21, 5, 7, 17, 14, 20),
        array([ 18.,   6.,  20., -16.,  24., -16., -54., -70., -19.]))]


        15 solution (too ugly)
        ((0, 21, 3, 5, 7, 35, 25, 18, 8, 11, 14, 33, 20, 12, 24), 
        array([ 84.66666667,  64.        ,  65.        ,  37.33333333, 
        23.33333333, 118.66666667,  26.33333333, 132.        ,
        80.66666667, 183.66666667,  44.        , -35.66666667,
        -25.        ,  43.66666667, -64.33333333,  18.66666667])
        '''

        # sols = is_linearly_separable_smallest_groups_first(
        #     mon_str_dict,
        #     state_matrix_1,
        #     y_1,
        #     starter_groups,
        #     max_threshold = max_threshhold,
        #     early_stop = True,
        # )

        if sols:
            # features_found = sols[0][0]
            # num_features_found = sum([len(subgroup) for subgroup in features_found])

            # if num_features_found < best_so_far:
            #     best_so_far = num_features_found
            #     print(num_features_found)
            #     print(features_found)
            print(sols)
            print(starter_indices)




if __name__ == '__main__':
    path = 'data/datasets/jsons/other.json'
    with open(path, 'r') as f: states = json.load(f)

    key_str = '''
    c1 e1 c2
    e2 c  e3
    c3 e4 c4
    '''
    print(key_str)

    # for move_of_interest in [0]:
        
    #     is_linear = find_small_linearly_seperable_dsets(
    #         states=states,
    #         move_of_interest=move_of_interest,
    #         max_degree=2,
    #     )
    
    # find_minimal_features_for_corners_and_edges()

    main()

    # build_solver()

    '''
    c1 e1 c2
    e2 c  e3
    c3 e4 c4

    1:

    -  2 c4   -  2 c3   +  6 c3c4 -  5 e3e4 -  5 e2e4 -  6  
    +  1 e4   +  2 c3c4 -  2 e3e4 -  2 e2e4 +  2 c1c2 -  2  
    +  3 c3c4 -  2 e3e4 -  2 e2e4 +  2 c2   +  2 c1   -  3  *

    0: 

    c1,c2,c3,c4,e1e2,e1e3,e2e4,e3e4,c1c2,c1c3,c2c4,c3c4, 'cc4', 'cc3', 'ce3'   [  0.  -2.   2.  18.  42.  26.  18.  44.   0.   0. -12. -16.  68. -16.  26. -17.]
    c1,c2,c3,c4,e1e2,e1e3,e2e4,e3e4,c1c2,c1c3,c2c4,c3c4, 'cc4', 'cc3', 'e1c'   [  0.  14.  12.  18.  28.  20.  20.  32.   0.   0. -12. -14.  40.  14. -16. -13.]

    
    the minimal amount of features (that include their symmetric variants) that can both linearly separate 0 and 1 is:

    20 
    ['c3', 'c4', 'c1', 'c2'],                                           # corners
    ['cc3', 'c2c', 'cc4', 'c1c'],                                       # corners + center
    ['e1c', 'e2c', 'ce3', 'ce4'],                                       # edges + center
    ['e4c4', 'c3e4', 'c2e3', 'e1c2', 'e2c3', 'e3c4', 'c1e2', 'c1e1'],   # all (corner, adjacent edge) pairs

    OR the even better

    20 
    ['c2', 'c1', 'c4', 'c3']          # corners
    ['c2c4', 'c3c4', 'c1c2', 'c1c3']  # adjacent corners
    ['e1e3', 'e3e4', 'e1e2', 'e2e4']  # adjacent edges
    ['c1c', 'cc3', 'c2c', 'cc4']      # corners * center
    ['e1c', 'ce4', 'e2c', 'ce3']      # edges * center
    [ 1.  0.  2.  1. -1. -1.  0.  0.  2.  4.  4.  2.  0.  1.  1.  8. -0.  2.  0.  2. -1.]
    [ 1.  1.  0.  0.  0.  2.  1.  0.  0. -2.  0. -2. -0. -0. -0. -0.  0. -0. -2. -2. -2.]

    Corner discriminator len: 8
    ['c2', 'c4', 'c3', 'e3e4', 'e1e2', 'cc4', 'e1c', 'ce3']
    [ 2. 10.  2. 18.  6. 44. -6. 12. -7.]
    ['c2', 'c4', 'c3', 'e3e4', 'e1e2', 'cc4', 'ce4', 'e2c']
    [ 2. 10.  2. 18.  6. 44. 12. -6. -7.]
    ['c2', 'c4', 'c3', 'e3e4', 'e1e2', 'cc4', 'ce4', 'ce3']
    [ 2.  8.  2. 14. 18. 62. 14. 14. -5.]
    ['c2', 'c4', 'c3c4', 'e3e4', 'e1e2', 'cc4', 'ce4', 'e2c']
    [ 2. 14. -2. 24.  8. 60. 16. -8. -9.]
    ['c2', 'c4', 'c3c4', 'e3e4', 'e1e2', 'cc4', 'ce4', 'ce3']
    [ 2.  8. -2. 12.  8. 42. 12.  6. -3.]
    ['c4', 'c3', 'c2c4', 'e3e4', 'e1e2', 'cc4', 'e1c', 'ce3']
    [14.  2. -2. 24.  8. 60. -8. 16. -9.]
    ['c4', 'c3', 'c2c4', 'e3e4', 'e1e2', 'cc4', 'ce4', 'ce3']
    [ 8.  2. -2. 12.  8. 42. 12.  6. -3.]

    ['c4', 'c2c4', 'c3c4', 'e3e4', 'e1e2', 'cc4', 'ce4', 'ce3']
    [ 3. -2. -2.  6.  4. 19.  6.  3. -2.] * 

    TODO filter discriminators further
    TODO figure out how to handle 0,1 positions

    '''
