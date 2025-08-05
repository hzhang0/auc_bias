import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import scipy.stats
import warnings
from matplotlib.patches import Patch

def condense_dict(k, d):
    new_d = {}
    for i in d:
        if isinstance(d[i], dict):
            new_dict = condense_dict(i, d[i])
            new_d = {**new_d, **{k + '_' + j: new_dict[j] for j in new_dict}}
        else:
            new_d[k + '_' + i] = d[i]
    return new_d


def series_to_dict(srs, prefix):
    return {prefix + i: srs[i] for i in srs.index}

def flatten_individual_res(x):
    x_new = {}
    for sset in x:
        x_new = {**x_new, **condense_dict(f'{sset}_overall', x[sset]['overall'])}
        x_new = {**x_new, **series_to_dict(pd.Series(x[sset]['min_attr']), f'{sset}_min_attr_')}
        x_new = {**x_new, **series_to_dict(pd.Series(x[sset]['max_attr']), f'{sset}_max_attr_')}
        x_new = {**x_new, **series_to_dict(pd.Series(x[sset]['max_gap']), f'{sset}_max_gap_')}
        x_new = {**x_new, **series_to_dict(pd.Series(x[sset]['signed_gap']), f'{sset}_signed_gap_')}
        x_new = {**x_new, **series_to_dict(pd.Series(x[sset]['ratio']), f'{sset}_ratio_')}
    return x_new       

# heatmap
def make_heatmap(y_var, y_label, auroc_df, auprc_df, prev_gap,
                auroc_label = 'AUROC Selection', auprc_label = 'AUPRC Selection',
                show_prev_ratio = False,
                auroc_lower_df = None,
                auroc_upper_df = None,
                auprc_lower_df = None,
                auprc_upper_df = None,
                subset_idx = None,
                n_series = None,
                 plot_args = {}):
    
    auroc_color = 'tab:red'
    auprc_color = 'dodgerblue'
    
    use_ebars = auroc_lower_df is not None
    plt.rcParams.update({'font.size': 14})

    new_order = prev_gap.sort_values().index
    if subset_idx is not None:
        new_order = subset_idx
        # new_order = [i for i in new_order if i in subset_idx]
    
    f = plt.figure(figsize=(len(new_order) * 2, 5))
    x_labels, vals, colors = [], [], []
    lowers, uppers = [], []

    for c, idx in enumerate(new_order):        
        vals += [
            auroc_df.loc[idx, y_var],
            auprc_df.loc[idx, y_var]
        ]
        
        if use_ebars:
            lowers += [
                auroc_df.loc[idx, y_var] - auroc_lower_df.loc[idx, y_var],
                auprc_df.loc[idx, y_var] - auprc_lower_df.loc[idx, y_var]
            ]
            
            uppers += [
                auroc_upper_df.loc[idx, y_var] - auroc_df.loc[idx, y_var],
                auprc_upper_df.loc[idx, y_var] - auprc_df.loc[idx, y_var]
            ]
            

        colors += [auroc_color, auprc_color]
        
        if subset_idx is None:
            label = idx[0] + ': ' + idx[1].lower()
        else:
            label = idx[0]
        if label.endswith('1'):
            label = label[:-1]        
        
        if n_series is not None:
            p = compare_spearman_r(auroc_df.loc[idx, y_var], n_series.loc[idx], auprc_df.loc[idx, y_var], n_series.loc[idx])[1]
            if p < 0.001:
                label += '***'
            elif p < 0.01:
                label += '**'    
            elif p < 0.05:
                label += '*'

        if show_prev_ratio:
            label = label + f'\n({prev_gap.loc[idx]:.1f})'
        x_labels.append(label)
        
        if c != len(new_order) - 1:
            vals.append(0.)
            lowers.append(0.)
            uppers.append(0.)
            colors.append('red')

    ax = plt.gca()
    x_pos = np.arange(len(vals))
    if use_ebars:
        cont = ax.bar(x_pos, vals, yerr = np.array([lowers, uppers]), align='center', alpha=1, ecolor='black', color=colors,
                      linewidth = [1 if i != 0 else 0 for i in vals],
                     edgecolor = 'k',
                     **plot_args)
    else:
        cont = ax.bar(x_pos, vals, align='center', alpha=1, ecolor='black', color=colors, 
                      linewidth = [1 if i != 0 else 0 for i in vals],
                      edgecolor = 'k',
                     **plot_args)
    ax.set_xticks(np.arange(2, len(colors) + 1, 3) - 1.5)
    ax.set_xticklabels(x_labels)
    ax.xaxis.set_ticks_position('none') 

    line_locs = np.arange(2, len(colors) + 1, 3)
    for i in line_locs[:-1]:
        ax.axvline(i, linestyle = '--', color = 'gray', linewidth = 1)

    # plt.xlabel("Dataset + Attribute")
    plt.ylabel(y_label)

    legend_elements = [Patch(facecolor=auroc_color,  label=auroc_label),
                      Patch(facecolor=auprc_color,  label=auprc_label)]
    
    ax.yaxis.grid(True, which='major')

    plt.legend(handles=legend_elements)
    return f

def spearman_ci(r, n):
    stderr = 1.0 / np.sqrt(n - 3)
    delta = 1.96 * stderr
    lower = np.tanh(np.arctanh(r) - delta)
    upper = np.tanh(np.arctanh(r) + delta)
    return (lower, upper)

def spearman_diff_ci(r1, n1, r2, n2, confidence = 0.95): # this is weird
    Z1 = np.arctanh(r1)
    Z2 = np.arctanh(r2)
    Z_diff = Z1 - Z2
    SE_diff = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
    Z_score = scipy.stats.norm.ppf(1 - (1 - confidence) / 2)  # two-tailed
    CI_Z = (Z_diff - Z_score * SE_diff, Z_diff + Z_score * SE_diff)
    CI_r = (np.tanh(CI_Z[0]), np.tanh(CI_Z[1]))
    return CI_r

def compare_spearman_r(r1, n1, r2, n2):
    Z1 = np.arctanh(r1)
    Z2 = np.arctanh(r2)
    SE1 = 1 / np.sqrt(n1 - 3)
    SE2 = 1 / np.sqrt(n2 - 3)
    Z_score = (Z1 - Z2) / np.sqrt(SE1**2 + SE2**2)
    
    # p-value from the Z-score
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(Z_score)))  # Two-tailed test
    
    return Z_score, p_value

def make_heatmap_single(data_df, prev_gap,
                show_prev_ratio = False,
                        plot_args = {}):
    
#     auroc_color = 'tab:red'
#     auprc_color = 'dodgerblue'
    
    plt.rcParams.update({'font.size': 14})

    new_order = prev_gap.sort_values().index
    
    f = plt.figure(figsize=(len(new_order) * 1.2, 5))
    x_labels, vals, colors = [], [], []
    lowers, uppers = [], []       

    for c, idx in enumerate(new_order):        
        vals += [
            data_df.loc[idx, 'diff']
        ]
    
        lowers += [
            data_df.loc[idx, 'diff'] - data_df.loc[idx, 'lower']
        ]

        uppers += [
            data_df.loc[idx, 'upper'] - data_df.loc[idx, 'diff']
        ]
            
        colors += ['tab:red']
        
        # if subset_idx is None:
        label = idx[0] + '\n' + idx[1].lower()
#         else:
#             label = idx[0]
        if label.endswith('1'):
            label = label[:-1]        

        if show_prev_ratio:
            label = label + f'\n({prev_gap.loc[idx]:.1f})'
        x_labels.append(label)
        
        if c != len(new_order) - 1:
            vals.append(0.)
            lowers.append(0.)
            uppers.append(0.)
            colors.append('red')

    ax = plt.gca()
    x_pos = np.arange(len(vals))
    cont = ax.bar(x_pos, vals, yerr = np.array([lowers, uppers]), align='center', alpha=1, ecolor='black', color=colors,
                  linewidth = [1 if i != 0 else 0 for i in vals],
                 edgecolor = 'k',
                 **plot_args)
    
    ax.set_xticks(np.arange(0, len(colors) + 1, 2))
    ax.set_xticklabels(x_labels)
    ax.xaxis.set_ticks_position('none') 

    line_locs = np.arange(1, len(colors) + 1, 2)
    for i in line_locs[:-1]:
        ax.axvline(i, linestyle = '--', color = 'gray', linewidth = 1)

    # plt.xlabel("Dataset + Attribute")
    plt.ylabel('Difference in Spearman $\\rho$\n(AUROC Gap vs. AUPRC $-$ \nAUROC Gap vs. AUROC)')
    
    ax.yaxis.grid(True, which='major')

    return f