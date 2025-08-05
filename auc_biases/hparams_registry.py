import numpy as np
import hashlib


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert name not in hparams
        random_state = np.random.RandomState(
            seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    _hparam('use_sensitive', False, lambda r: r.choice([True, False]))

    if algorithm == 'lr':
        _hparam('clf__C', 0.1, lambda r: 10**r.uniform(-5, 1))
        _hparam('clf__penalty', 'l1', lambda r: r.choice(['l1', 'l2']))
    elif algorithm == 'svm':
        _hparam('clf__C', 0.1, lambda r: 10**r.uniform(-5, 1))
        _hparam('clf__kernel', 'rbf', lambda r: r.choice(['linear', 'poly', 'rbf', 'sigmoid']))
    elif algorithm == 'xgb':
        _hparam('clf__max_depth', 5, lambda r: r.randint(1, 10))
        _hparam('clf__learning_rate', 0.3, lambda r: r.uniform(0.01, 0.3))
        _hparam('clf__n_estimators', 100, lambda r: r.randint(50, 1000))
        _hparam('clf__min_child_weight', 1, lambda r: r.randint(1, 10))
    elif algorithm == 'rf':
        _hparam('clf__max_depth', 5, lambda r: r.randint(1, 10))
        _hparam('clf__n_estimators', 100, lambda r: r.randint(50, 1000))
        _hparam('clf__min_samples_split', 2, lambda r: r.randint(2, 20))
        _hparam('clf__min_samples_leaf', 1, lambda r: r.randint(1, 20))
    elif algorithm == 'nn':
        _hparam('clf__hidden_layer_sizes', (100,), lambda r: r.choice([(50,), (100,), (50, 100), (100, 50), (100, 100)]))
        _hparam('clf__activation', 'relu', lambda r: r.choice(['logistic', 'tanh', 'relu']))
        _hparam('clf__alpha', 0.0001, lambda r: 10**r.uniform(-7, -1))
    elif algorithm == 'nn_torch':
        _hparam('clf__max_epochs', 30, lambda r: r.randint(10, 100))
        _hparam('clf__batch_size', 128, lambda r: int(2**r.randint(5, 11)))
        _hparam('clf__lr', 0.01, lambda r: 10**r.uniform(-4, 1))
        _hparam('clf__mlp_depth', 3, lambda r: r.randint(2, 5))
        _hparam('clf__mlp_width', 128, lambda r: r.randint(32, 256))
        _hparam('clf__mlp_dropout', 0.1, lambda r: r.uniform(0, 0.5))
    elif algorithm == 'knn':
        _hparam('clf__n_neighbors', 5, lambda r: r.randint(1, 20))
        _hparam('clf__weights', 'uniform', lambda r: r.choice(['uniform', 'distance']))
    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}

def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}