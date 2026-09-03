## 
import json
import itertools
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from scipy.optimize import linprog
from scipy.special import binom

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
    # for grou in sorted(all_monomial_groups, key=len):
    #     print(grou)
    # input()
    return feature_state_matrix, all_monomials, all_monomial_groups

    
def filter_feature_state_matrix(states, feature_state_matrix, move_of_interest):
    '''
    
    '''
    y = []
    rows_to_keep = []
    for idx, (board, moves) in enumerate(states.items()):
        # if move_of_interest in moves and len(moves) == 1:
        if move_of_interest in moves and len(moves) == 1 and board.count(' ') != len(moves):
            rows_to_keep.append(idx)
            y.append(1)
        
        elif move_of_interest not in moves and board[move_of_interest] == ' ':
            rows_to_keep.append(idx)
            y.append(-1)

    return feature_state_matrix[rows_to_keep], y


def is_linearly_separable_recurse(
        filtered_feature_state_matrix,
        y,
        column_indices,
        sols_found,
        memoize
    ):
    '''
    
    '''
    state_tuple = tuple(column_indices)
    if state_tuple in memoize: return
    memoize.add(state_tuple)

    is_lin_sep, res_x = is_linearly_separable(filtered_feature_state_matrix[:, column_indices], y)

    if is_lin_sep:
        first_sol_indices = sols_found[0][0]
        # first_sol_indices = [name for group in sols_found[0][0] for name in group]
        # input(sols_found)
        # 
        if len(column_indices) < len(first_sol_indices): 
            sols_found[:] = [(column_indices, res_x)]
            print(f'{len(column_indices)} {column_indices} {res_x}')
        elif len(column_indices) == len(first_sol_indices):
            sols_found.append((column_indices, res_x))
            print(f'{len(column_indices)} {column_indices} {res_x}')
    else:
        return

    # try to remove one of the columns
    for i in range(len(column_indices)):
        reduced_column_indices = column_indices.copy()
        reduced_column_indices.pop(i)

        is_linearly_separable_recurse(
            filtered_feature_state_matrix,
            y,
            reduced_column_indices,
            sols_found,
            memoize=memoize
        )

def is_linearly_separable_k_subsets(
        state_matrix,
        y,
        column_indices,
        k
    ): 
        print(f'k = {k}')
        sols = []
        
        prog = tqdm(itertools.combinations(column_indices, k), total=binom(len(column_indices), k))
        prog.set_description(f'k={k} {len(sols)} solutions found.')
        for column_indices_subset in prog:
            is_lin_sep, res_x = is_linearly_separable(state_matrix[:, column_indices_subset], y)
            if is_lin_sep:
                sols.append((column_indices_subset, res_x))
                prog.set_description(f'k={k} {len(sols)} solutions found.')
                
        return sols

def is_linearly_separable_k_subsets_grouped(
        mon_str_dict,
        state_matrix_0,
        state_matrix_1,
        y_0,
        y_1,
        column_groups,
        k,
    ):    
        sols = []
        current_best = float('inf')
        
        # prog = tqdm(itertools.combinations(column_groups, k), total=binom(len(column_groups), k))
        # prog.set_description(f'k={k} {len(sols)} solutions found.')

        for column_groups_subset in valid_subsets(column_groups, max_threshold=20):
            column_indices_subset = [mon_str_dict[name] for group in column_groups_subset for name in group ]
            is_lin_sep_0, res_x_0 = is_linearly_separable(state_matrix_0[:, column_indices_subset], y_0)
            is_lin_sep_1, res_x_1 = is_linearly_separable(state_matrix_1[:, column_indices_subset], y_1)
            is_lin_sep = is_lin_sep_0 and is_lin_sep_1

            if is_lin_sep:
                if len(column_indices_subset) < current_best:
                    current_best = len(column_indices_subset)
                    sols = [(column_groups_subset, res_x_0, res_x_1)]
                    print(f'### new best: {current_best} ###\n{(column_groups_subset, res_x_0, res_x_1)}')
                elif len(column_indices_subset) == current_best:
                    sols.append((column_groups_subset, res_x_0, res_x_1))
                # prog.set_description(f'k={k} {len(sols)} {current_best}-solutions found.')
                
        return sols

def find_small_linearly_seperable_dsets(states, move_of_interest: int, max_degree: int = 3):
    '''
    
    '''
    print(f'Move of interest: {move_of_interest}')

    # create single feature state matrix
    feature_state_matrix, all_monomials, all_monomials_groups = build_feature_state_matrix(states, max_degree=max_degree, move_of_interest=move_of_interest)

    mon_str_dict = {monomial_to_feature_str(mon):idx for idx, mon in enumerate(all_monomials)} | {idx:monomial_to_feature_str(mon) for idx, mon in enumerate(all_monomials)}

    # filter rows, remove states that will not affect larger net
    filtered_feature_state_matrix, y = filter_feature_state_matrix(states, feature_state_matrix, move_of_interest)

    # filter columns, 
    nonzero_indices = np.any(filtered_feature_state_matrix != 0, axis=0)
    nonzero_filtered_feature_state_matrix = filtered_feature_state_matrix[:, nonzero_indices]

    # return seperator plane of all features with products up to max_degree
    is_lin_sep, res_x = is_linearly_separable(nonzero_filtered_feature_state_matrix, y)
    if not is_lin_sep:
        print('Not linearly separable!')
        return is_lin_sep

    # iteratively find planes that involve less features
    # reduced_column_indices = [idx for idx, val in enumerate(res_x[:-1]) if val != 0]
    # memoize = set()
    # sols_found = [(reduced_column_indices, res_x)]
    # is_linearly_separable_recurse(
    #     nonzero_filtered_feature_state_matrix,
    #     y,
    #     column_indices=reduced_column_indices,
    #     sols_found=sols_found,
    #     memoize=memoize
    # )

    # k = 6
    # sols = is_linearly_separable_k_subsets(
    #     nonzero_filtered_feature_state_matrix,
    #     y,
    #     column_indices=reduced_column_indices,
    #     k=k
    # )
    
    # sols = is_linearly_separable_from_starter(
    #     nonzero_filtered_feature_state_matrix,
    #     y,

    #     all_monomials,
    # )

    # filter_index_map = np.nonzero(nonzero_indices)[0]

    # mon_strs = [monomial_to_feature_str(mon) for mon in all_monomials ]
    # for idx, mon_str in enumerate(mon_strs):
    #     print(f'{idx}: {mon_str}')
    

    # for (sol_feature_idx, sol_coefs) in sols: 
    #     temp = [int(filter_index_map[i]) for i in sol_feature_idx]
    #     terms = [mon_strs[mon_code] for mon_code in temp] + [' ']
    #     final_str = ' '
    #     for coef, term in zip(sol_coefs, terms):
    #         coef_term, coef_sign = abs(int(coef)), int(coef)//abs(int(coef))
    #         sign_dict = {1:'+',-1:'-'}
    #         final_str += f'{sign_dict[coef_sign]} {coef_term:>2}{term} '
    #     print(final_str)

    # k = 3
    # print(f'checking k={k}')
    # for to_add in itertools.combinations(range(len(all_monomials)), k):
    #     all_new = True
    #     for thing in to_add:
    #         if thing in starter_indices:
    #             all_new = False
    #             break
    #     if all_new:
    #         temp = starter_indices.copy()
    #         for i in to_add:
    #             temp.append(i)
    #         is_lin_sep, res_x = is_linearly_separable(filtered_feature_state_matrix[:, temp], y)
    #         if is_lin_sep:
    #             print(f'{[mon_str_dict[i] for i in to_add]}   {res_x}')


    return is_lin_sep


def is_linearly_separable_recurse_grouped(
        mon_str_dict,
        filtered_feature_state_matrix,
        y,
        column_groups,
        sols_found,
        memoize
    ):
    '''
    
    '''
    # degroup for convenience
    column_indices = [mon_str_dict[name] for group in column_groups for name in group]

    #
    state_tuple = tuple(column_indices)
    if state_tuple in memoize: return
    memoize.add(state_tuple)

    # 
    is_lin_sep, res_x = is_linearly_separable(filtered_feature_state_matrix[:, column_indices], y)

    if is_lin_sep:
        # unpack first solution to get length
        first_sol_indices = [name for group in sols_found[0][0] for name in group]
        # 
        if len(column_indices) < len(first_sol_indices): 
            sols_found[:] = [(column_groups, res_x)]
            # print(f'{len(column_indices)} {column_indices} {res_x_0} | {res_x_1}')
            print(f'## {len(column_indices)} ## {[name for group in column_groups for name in group]}')
            print(f'{res_x}')
        elif len(column_indices) == len(first_sol_indices):
            sols_found.append((column_groups, res_x))
            # print(f'{len(column_indices)} {column_indices} {res_x_0} | {res_x_1}')
            print(f'## {len(column_indices)} ## {[name for group in column_groups for name in group]}')
            print(f'{res_x}')
    else:
        return

    # try to remove one of the columns
    for i in range(len(column_groups)):
        reduced_column_groups = deepcopy(column_groups)
        reduced_column_groups.pop(i)

        is_linearly_separable_recurse_grouped(
            mon_str_dict,
            filtered_feature_state_matrix,
            y,
            reduced_column_groups,
            sols_found,
            memoize=memoize
        )

def is_linearly_separable_recurse_grouped_both(
        mon_str_dict,
        filtered_feature_state_matrix_0,
        filtered_feature_state_matrix_1,
        y_0,
        y_1,
        column_groups,
        sols_found,
        memoize
    ):
    '''
    
    '''
    # degroup for convenience
    column_indices = [mon_str_dict[name] for group in column_groups for name in group]

    #
    state_tuple = tuple(column_indices)
    if state_tuple in memoize: return
    memoize.add(state_tuple)

    # 
    is_lin_sep_0, res_x_0 = is_linearly_separable(filtered_feature_state_matrix_0[:, column_indices], y_0)
    is_lin_sep_1, res_x_1 = is_linearly_separable(filtered_feature_state_matrix_1[:, column_indices], y_1)
    is_lin_sep = is_lin_sep_0 and is_lin_sep_1

    if is_lin_sep:
        # unpack first solution to get length
        first_sol_indices = [name for group in sols_found[0][0] for name in group]
        # 
        if len(column_indices) < len(first_sol_indices): 
            sols_found[:] = [(column_groups, res_x_0, res_x_1)]
            # print(f'{len(column_indices)} {column_indices} {res_x_0} | {res_x_1}')
            print(f'## {len(column_indices)} ## {[name for group in column_groups for name in group]}')
        elif len(column_indices) == len(first_sol_indices):
            sols_found.append((column_groups, res_x_0, res_x_1))
            # print(f'{len(column_indices)} {column_indices} {res_x_0} | {res_x_1}')
            print(f'## {len(column_indices)} ## {[name for group in column_groups for name in group]}')
    else:
        return

    # try to remove one of the columns
    for i in range(len(column_groups)):
        reduced_column_groups = deepcopy(column_groups)
        reduced_column_groups.pop(i)

        is_linearly_separable_recurse_grouped_both(
            mon_str_dict,
            filtered_feature_state_matrix_0,
            filtered_feature_state_matrix_1,
            y_0,
            y_1,
            reduced_column_groups,
            sols_found,
            memoize=memoize
        )


def find_minimal_features_for_corners_and_edges():
    ''' search for total collection of features for moves 0 and 1, removing features in symmetric equiv classes '''

    # create single feature state matrix
    feature_state_matrix, all_monomials, all_monomial_groups = build_feature_state_matrix(states, max_degree=3)

    mon_str_dict = {monomial_to_feature_str(mon):idx for idx, mon in enumerate(all_monomials)} | {idx:monomial_to_feature_str(mon) for idx, mon in enumerate(all_monomials)}

    # for key, val in mon_str_dict.items(): print(f'{key}: {val}')

    # filter rows, remove states that will not affect larger net
    filtered_feature_state_matrix_0, y_0 = filter_feature_state_matrix(states, feature_state_matrix, 0)
    filtered_feature_state_matrix_1, y_1 = filter_feature_state_matrix(states, feature_state_matrix, 1)

    starter_groups = [
        ['c2', 'c1', 'c4', 'c3'],          # corners
        ['c2c4', 'c3c4', 'c1c2', 'c1c3'],  # adjacent corners
        ['e1e3', 'e3e4', 'e1e2', 'e2e4'],  # adjacent edges
        ['c1c', 'cc3', 'c2c', 'cc4'],      # corners * center
        ['e1c', 'ce4', 'e2c', 'ce3'],      # edges * center
    ]
    # starter_indices = [mon_str_dict[name] for group in all_monomial_groups for name in group ]
    starter_indices = [mon_str_dict[name] for group in starter_groups for name in group ]

    # 
    is_lin_sep_0, res_x_0 = is_linearly_separable(filtered_feature_state_matrix_0[:, starter_indices], y_0)
    is_lin_sep_1, res_x_1 = is_linearly_separable(filtered_feature_state_matrix_1[:, starter_indices], y_1)
    is_lin_sep = is_lin_sep_0 and is_lin_sep_1
    if not is_lin_sep:
        print(f'Starter is NOT linearly separable! 0: {is_lin_sep_0} 1: {is_lin_sep_1}')
        return is_lin_sep_0, is_lin_sep_1
    else:
        print(f'Starter is linearly separable!')


    ## iteratively find planes that involve less features
    # memoize = set()
    # sols_found = [(all_monomial_groups, res_x_0, res_x_1)]
    # is_linearly_separable_recurse_grouped_both(
    #     mon_str_dict,
    #     filtered_feature_state_matrix_0,
    #     filtered_feature_state_matrix_1,
    #     y_0,
    #     y_1,
    #     column_groups=all_monomial_groups,
    #     sols_found=sols_found,
    #     memoize=memoize
    # )

    

    # k = 5
    # sols_found = is_linearly_separable_k_subsets_grouped(
    #     mon_str_dict,
    #     filtered_feature_state_matrix_0,
    #     filtered_feature_state_matrix_1,
    #     y_0,
    #     y_1,
    #     all_monomial_groups,
    #     k,
    # )

    # memoize = set()
    # sols_found = [(starter_indices, res_x_0)]
    # is_linearly_separable_recurse(
    #     filtered_feature_state_matrix_0,
    #     y_0,
    #     starter_indices,
    #     sols_found,
    #     memoize
    # )

    sols_found = is_linearly_separable_k_subsets(
        filtered_feature_state_matrix_0,
        y_0,
        starter_indices,
        k=8
    )
    if len(sols_found) > 0:
        print('### Solutions ###')
        for sol in sols_found:
            sol_indices, sol_for_0 = sol
            sol_name = [mon_str_dict[idx] for idx in sol_indices]
            print(f'{len(sol_indices)} {sol_name}')
            print(sol_for_0)
    
    
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
    
    find_minimal_features_for_corners_and_edges()

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
