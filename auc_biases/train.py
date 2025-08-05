import argparse
import collections
import json
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from auc_biases import TabularDataset, hparams_registry
from auc_biases.metrics import eval_metrics
from auc_biases.models import train_clf

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required = True)
parser.add_argument('--dataset', type=str, choices = list(TabularDataset.dataset_params.keys()), required = True)
parser.add_argument('--attribute', type=int, choices=[0, 1], help = "All datasets have two possible attibutes (e.g. sex and race)", required = True)
parser.add_argument('--algorithm', type=str, default="lr", choices=['lr', 'xgb', 'rf', 'svm', 'nn', 'knn', 'nn_torch'])
parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for random hparams (0 for "default hparams")')
parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict', required = False)
parser.add_argument('--seed', type=int, default=0, help='Seed for everything else, most notably the dataset split')
parser.add_argument('--balance_groups', action = 'store_true', help="balance groups in both training and test by subsampling")
parser.add_argument('--enforce_prevalence_ratio', action = 'store_true', help="whether to subset samples to enforce a prevalence ratio")
parser.add_argument('--prevalence_ratio', type = float, help="if greater than actual, subset y=1 samples from lower prev group, \
    else subset y=1 samples from higher prevalence group")
parser.add_argument('--higher_prev_group_weight', default = 1., type = float, help="sample weight for the higher prevalence group")
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents = True, exist_ok = True)

print('Args:')
for k, v in sorted(vars(args).items()):
    print('\t{}: {}'.format(k, v))

if args.hparams_seed == 0:
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
else:
    hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, hparams_registry.seed_hash(args.hparams_seed))
if args.hparams:
    hparams.update(json.loads(args.hparams))

print('HParams:')
for k, v in sorted(hparams.items()):
    print('\t{}: {}'.format(k, v))

with open(os.path.join(output_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

random.seed(args.seed)
np.random.seed(args.seed)

dataset = TabularDataset.Dataset(args.dataset, use_sensitive = hparams['use_sensitive'])
X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test = dataset.get_data(seed = args.seed, force_resplit = True,
                                                                                           balance_groups = args.balance_groups, attribute_index=args.attribute,
                                                                                           enforce_prevalence_ratio = args.enforce_prevalence_ratio,
                                                                                           prevalence_ratio = args.prevalence_ratio)

attr_col = dataset.sensitive_attributes[args.attribute]
g_train, g_val, g_test = g_train[attr_col], g_val[attr_col], g_test[attr_col]

clf, pred_train, pred_val, pred_test = train_clf(args.algorithm, X_train, X_val, X_test, y_train, g_train, model_hparams = {i[5:]:hparams[i] for i in hparams if i.startswith('clf__')},
                                   cat_cols = TabularDataset.dataset_params[args.dataset].categorical_columns, 
                                   higher_prev_group_weight = args.higher_prev_group_weight)

final_results = {attr_col: {
    'tr': eval_metrics(pred_train, y_train, g_train.values),
    'va': eval_metrics(pred_val, y_val, g_val.values),
    'te': eval_metrics(pred_test, y_test, g_test.values)
}}

pickle.dump(final_results, open(os.path.join(output_dir, 'final_results.pkl'), 'wb'))

with (output_dir/'done').open('w') as f:
    f.write('done')
