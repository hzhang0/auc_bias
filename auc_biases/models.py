import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
from collections import Counter
from auc_biases.torch_nn import MLPWithWeights, MLP
import torch

warnings.filterwarnings("ignore")

def add_scaler(clf, all_col_indices, cat_col_indices = []):
    steps = [
        ('scaler', StandardScaler()),
        ('clf', clf)
    ]
    if len(cat_col_indices) > 0:
        numeric_features=list(set(all_col_indices)-set(cat_col_indices))
        numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, cat_col_indices)])

        steps=[
               ('scaler', preprocessor),
               ('clf', clf)
            ]

    return Pipeline(steps)


def get_model(x, y, g, model_name, model_hparams, cat_cols = [], higher_prev_group_weight= 1):
    if len(cat_cols):
        assert isinstance(x, pd.DataFrame)
        cat_col_indices = [list(x.columns).index(i) for i in cat_cols]
    else:
        cat_col_indices = []

    if isinstance(x, pd.DataFrame):
        x = x.values

    if y.ndim == 2:
        y = y.squeeze()

    weights = np.ones(y.shape)
    prev_groups = {}
    for i in np.unique(g):
        y_g = y[g == i]
        prev_groups[i] = y.sum()/len(y)
    max_prev_group = pd.Series(prev_groups).idxmax()
    weights[g == max_prev_group] = higher_prev_group_weight

    if model_name == 'lr':
        clf = add_scaler(
            LogisticRegression(solver = 'liblinear', **model_hparams), cat_col_indices = cat_col_indices, all_col_indices=list(range(x.shape[1])) )
    elif model_name == 'nn':
        clf = add_scaler(
            MLPClassifier(**model_hparams), cat_col_indices = cat_col_indices, all_col_indices=list(range(x.shape[1])) )
    elif model_name == 'svm':
        clf = add_scaler(SVC(probability=True, **model_hparams), all_col_indices=list(range(x.shape[1])), cat_col_indices = cat_col_indices)
    elif model_name == 'xgb':
        clf = Pipeline([('clf', XGBClassifier(**model_hparams))])
    elif model_name == 'rf':
        clf = Pipeline([('clf', RandomForestClassifier(**model_hparams))])
    elif model_name == 'nn_torch':
        clf = add_scaler(MLPWithWeights(debug = False, **model_hparams),
                              all_col_indices=list(range(x.shape[1])), cat_col_indices = cat_col_indices)
    else:
        raise NotImplementedError(model_name)

    if model_name not in ['nn', 'knn']: 
        clf.fit(x, y, clf__sample_weight = weights)  
    else:
        clf.fit(x, y)

    return clf

def train_clf(model_name, x_train, x_val,
              x_test, y_train, g_train,
              model_hparams, cat_cols = [],
              higher_prev_group_weight = 1):
    clf = get_model(x_train, y_train, g_train, model_name, model_hparams, cat_cols = cat_cols,
                    higher_prev_group_weight = higher_prev_group_weight)
    if isinstance(x_test, pd.DataFrame):
        x_train = x_train.values
        x_val = x_val.values
        x_test = x_test.values

    return clf, clf.predict_proba(x_train)[:, 1], clf.predict_proba(x_val)[:, 1], clf.predict_proba(x_test)[:, 1]