import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from auc_biases.datasets import params_compas,params_adult, params_lsac, params_mimic_tab
from auc_biases.utils import random_subsample
import numpy as np

dataset_params = {
    'adult': params_adult,
    'lsac': params_lsac,
    'mimic': params_mimic_tab,
    'compas':params_compas,
}

class Dataset:
    def __init__(self, ds_name, use_sensitive = False):
        opts = dataset_params[ds_name]

        self.ds_name = ds_name
        self.link = opts.link
        self.columns = opts.columns
        self.train_cols = opts.train_cols
        self.label = opts.label
        self.sensitive_attributes = opts.sensitive_attributes
        self.use_sensitive = use_sensitive
        self.already_split = opts.already_split
        self.categorical_columns = opts.categorical_columns
        self.has_header = opts.has_header

        if self.use_sensitive:
            for i in self.sensitive_attributes:
                if i not in self.train_cols:
                    self.train_cols.append(i)
        else:
            for i in self.sensitive_attributes:
                if i in self.train_cols:
                    self.train_cols.remove(i)

    def get_data(self, seed = 0, force_resplit = False, balance_groups = False, attribute_index = 0,
                 enforce_prevalence_ratio = False, prevalence_ratio = None):
        df = pd.read_csv(
            self.link,
            header=0 if self.has_header else None)
        if not self.has_header:
            df.columns = self.columns

        label = self.label
        if isinstance(df[label].iloc[0], str):
            assert self.ds_name == 'adult'
            df[label] = df[label].map({
                ' <=50K': 0,
                ' >50K': 1
            })
            df = df[df['Race'].isin([' White', ' Black'])]
        elif self.ds_name == 'compas':
            df = df[df.race.isin(['African-American', 'Caucasian'])]
        elif self.ds_name == 'lsac':
            df = df[df.race1.isin([0, 3])] # 0 = White, 3 = Black
        elif self.ds_name == 'mimic':
            df = df[df.Race.isin(['white', 'black'])]

        train_cols = self.train_cols
        cat_cols_all=self.categorical_columns + [i for i in self.sensitive_attributes if isinstance(df[i].iloc[0], str)]
                
        for col in cat_cols_all:
            enc = OrdinalEncoder()
            df[col] = enc.fit_transform(
                df[col].values.reshape(-1, 1))
            
        if enforce_prevalence_ratio:
            g = df[self.sensitive_attributes[attribute_index]]
            prev_groups = {}
            for i in np.unique(g):
                prev_groups[i] = df.loc[g == i, label].sum()/(g == i).sum()
            max_prev_group = pd.Series(prev_groups).idxmax()
            min_prev_group = pd.Series(prev_groups).idxmin()

            current_ratio = prev_groups[max_prev_group] / prev_groups[min_prev_group]

            if prevalence_ratio > current_ratio: # subset y=1 samples from lower prev group
                desired_prev_lower_prev = prev_groups[max_prev_group] / prevalence_ratio
                df_lower = df[g == min_prev_group]
                n_pos_lower = int(desired_prev_lower_prev * (df_lower[label] == 0).sum()/(1 - desired_prev_lower_prev ))
                df_lower = pd.concat((df_lower[df_lower[label] == 0],
                                      df_lower[df_lower[label] == 1].sample(
                                          n = n_pos_lower,
                                          replace = False, random_state = seed)), ignore_index = False
                )
                df = pd.concat((df_lower, df[g != min_prev_group]), ignore_index = True)
            else: # subset y=1 samples from higher prev group
                desired_prev_higher_prev = prev_groups[min_prev_group] * prevalence_ratio
                df_upper = df[g == max_prev_group]
                n_pos_upper = int(desired_prev_higher_prev * (df_upper[label] == 0).sum()/(1 - desired_prev_higher_prev))
                df_upper = pd.concat((df_upper[df_upper[label] == 0],
                                      df_upper[df_upper[label] == 1].sample(
                                          n = n_pos_upper,
                                          replace = False, random_state = seed)), ignore_index = False
                )
                df = pd.concat((df_upper, df[g != max_prev_group]), ignore_index = True)
                            
        if balance_groups:
            df = df.loc[random_subsample(df.index, df[self.sensitive_attributes[attribute_index]], seed = seed)]
        
        if not self.already_split or force_resplit:
            X = df[train_cols]
            y = df[label].values
            
            X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
                X, y, df[self.sensitive_attributes], test_size=0.5, random_state=seed, stratify = df[self.sensitive_attributes[attribute_index]])
            X_val, X_test, y_val, y_test, g_val, g_test = train_test_split(
                X_test, y_test, g_test, test_size=0.5, random_state=seed, stratify = g_test[self.sensitive_attributes[attribute_index]])
        else:
            df_train = df[df.fold_id =='train']
            df_val = df[df.fold_id =='eval']
            df_test = df[df.fold_id =='test']

            X_train, X_val, X_test = df_train[train_cols], df_val[train_cols], df_test[train_cols]
            y_train, y_val, y_test = df_train[label].values, df_val[label].values, df_test[label].values
            g_train, g_val, g_test = df_train[self.sensitive_attributes], df_val[self.sensitive_attributes], df_test[self.sensitive_attributes]
            
        return X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test