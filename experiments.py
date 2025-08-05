import os
import json
from pathlib import Path
from itertools import product


def combinations_base(grid):
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))

def combinations(grid):
    sub_exp_names = set()
    args = []
    for i in grid:
        if isinstance(grid[i], dict):
            for j in grid[i]:
                sub_exp_names.add(j)
    if len(sub_exp_names) == 0:
        return combinations_base(grid)

    for i in grid:
        if isinstance(grid[i], dict):
            assert set(list(grid[i].keys())) == sub_exp_names, f'{i} does not have all sub exps ({sub_exp_names})'
    for n in sub_exp_names:
        sub_grid = grid.copy()
        for i in sub_grid:
            if isinstance(sub_grid[i], dict):
                sub_grid[i] = sub_grid[i][n]
        args += combinations_base(sub_grid)
    return args


def get_hparams(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment]().get_hparams()


def get_script_name(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].fname

class vary_group_weight_with_seeds():
    fname = 'train'
    def __init__(self):
        self.hparams = {
            'dataset': ['adult', 'lsac', 'mimic', 'compas'],
            'algorithm': ['xgb'],
            'attribute': [0, 1],
            'hparams_seed': list(range(50)),
            'balance_groups': [True],
            'seed': list(range(20)), 
            'enforce_prevalence_ratio': [False],
            'higher_prev_group_weight': [1, 2, 3, 4, 5, 10, 15, 20, 25, 50]
        }

    def get_hparams(self):
        return combinations(self.hparams)


class vary_group_weight_with_seeds_nn():
    fname = 'train'
    def __init__(self):
        self.hparams = {
            'dataset': ['adult', 'lsac', 'mimic', 'compas'],
            'algorithm': ['nn_torch'],
            'attribute': [0, 1],
            'hparams_seed': list(range(25)),
            'balance_groups': [True],
            'seed': list(range(5)), 
            'enforce_prevalence_ratio': [False],
            'higher_prev_group_weight': [1, 2, 3, 4, 5, 10, 15, 20, 25, 50]
        }

    def get_hparams(self):
        return combinations(self.hparams)
    
