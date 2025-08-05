import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, average_precision_score,
                             balanced_accuracy_score, recall_score, brier_score_loss, log_loss, classification_report)
import netcal.metrics


def eval_metrics(preds, targets, attributes, thress=[0.5], thress_suffix=['_50'], add_arrays=False):
    label_set = np.unique(targets)

    preds_rounded = {suffix: preds >= thres if preds.squeeze().ndim == 1 else preds.argmax(1)
                     for thres, suffix in zip(thress, thress_suffix)}
    res = {}
    res['overall'] = prob_metrics(targets, preds, label_set)
    for thres, suffix in zip(thress, thress_suffix):
        res['overall'] = {**res['overall'], **binary_metrics(targets, preds_rounded[suffix], label_set, suffix=suffix)}

    res['per_attribute'] = {}
    res['per_class'] = {}

    # per attribute results
    prevalence_per_group = {}
    for a in np.unique(attributes):
        mask = attributes == a
        res['per_attribute'][int(a)] = prob_metrics(targets[mask], preds[mask], label_set)
        for thres, suffix in zip(thress, thress_suffix):
            res['per_attribute'][int(a)] = {**res['per_attribute'][int(a)], **binary_metrics(
                targets[mask], preds_rounded[suffix][mask], label_set, suffix=suffix)}
        prevalence_per_group[int(a)] = res['per_attribute'][int(a)]['prevalence'+suffix]

    # per class binary results
    for thres, suffix in zip(thress, thress_suffix):
        classes_report = classification_report(targets, preds_rounded[suffix], output_dict=True, zero_division=0.)
        res['overall'][f'macro_avg_{suffix}'] = classes_report['macro avg']
        res['overall'][f'weighted_avg_{suffix}'] = classes_report['weighted avg']
        for y in np.unique(targets):
            res['per_class'][int(y)] = {f'{i}{suffix}': classes_report[str(y)][i] for i in classes_report[str(y)]}

    # per class AUROC
    if preds.squeeze().ndim == 1:  # 2 classes
        res['per_class'][1]['AUROC'] = roc_auc_score(targets, preds, labels=[0, 1])
        res['per_class'][0]['AUROC'] = res['per_class'][1]['AUROC']
    else:
        for y in np.unique(targets):
            new_label = targets == y
            new_preds = preds[:, int(y)]
            res['per_class'][int(y)]['AUROC'] = roc_auc_score(new_label, new_preds, labels=[0, 1])

    res['min_attr'] = pd.DataFrame(res['per_attribute']).min(axis=1).to_dict()
    res['max_attr'] = pd.DataFrame(res['per_attribute']).max(axis=1).to_dict()
    res['max_gap'] = (pd.DataFrame(res['per_attribute']).max(axis=1) - pd.DataFrame(res['per_attribute']).min(axis=1)).to_dict()

    max_prev_group = pd.Series(prevalence_per_group).idxmax()
    min_prev_group = pd.Series(prevalence_per_group).idxmin()
    res['signed_gap'] = (pd.Series(res['per_attribute'][max_prev_group]) - pd.Series(res['per_attribute'][min_prev_group])).to_dict()
    res['ratio'] = (pd.Series(res['per_attribute'][max_prev_group]) / pd.Series(res['per_attribute'][min_prev_group])).to_dict()

    if add_arrays:
        res['y'] = targets
        res['a'] = attributes
        res['preds'] = preds

    return res


def binary_metrics(targets, preds, label_set=[0, 1], suffix='', return_arrays=False):
    if len(targets) == 0:
        return {}

    res = {
        'accuracy': accuracy_score(targets, preds),
        'n_samples': len(targets)
    }

    if len(label_set) == 2:
        CM = confusion_matrix(targets, preds, labels=label_set)

        res['TN'] = CM[0][0].item()
        res['FN'] = CM[1][0].item()
        res['TP'] = CM[1][1].item()
        res['FP'] = CM[0][1].item()

        res['error'] = res['FN'] + res['FP']

        if res['TP'] + res['FN'] == 0:
            res['TPR'] = 0
            res['FNR'] = 1
        else:
            res['TPR'] = res['TP']/(res['TP']+res['FN'])
            res['FNR'] = res['FN']/(res['TP']+res['FN'])

        if res['FP'] + res['TN'] == 0:
            res['FPR'] = 1
            res['TNR'] = 0
        else:
            res['FPR'] = res['FP']/(res['FP']+res['TN'])
            res['TNR'] = res['TN']/(res['FP']+res['TN'])

        res['pred_prevalence'] = (res['TP'] + res['FP']) / res['n_samples']
        res['prevalence'] = (res['TP'] + res['FN']) / res['n_samples']
    else:
        CM = confusion_matrix(targets, preds, labels=label_set)
        res['TPR'] = recall_score(targets, preds, labels=label_set, average='macro', zero_division=0.)

    if len(np.unique(targets)) > 1:
        res['balanced_acc'] = balanced_accuracy_score(targets, preds)

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds

    return {f"{i}{suffix}": res[i] for i in res}


def prob_metrics(targets, preds, label_set, return_arrays=False):
    if len(targets) == 0:
        return {}

    res = {
        'BCE': log_loss(targets, preds, eps=1e-6, labels=label_set),
        'ECE': netcal.metrics.ECE().measure(preds, targets)
    }

    if len(set(targets)) > 2:
        # happens when you predict a class, but there are no samples with that class in the dataset
        try:
            res['AUROC'] = roc_auc_score(targets, preds, multi_class='ovr', labels=label_set)
        except:
            res['AUROC'] = roc_auc_score(targets, preds, multi_class='ovo', labels=label_set)
    elif len(set(targets)) == 2:
        res['AUROC'] = roc_auc_score(targets, preds, labels=label_set)
    elif len(set(targets)) == 1:
        res['AUROC'] = None

    if len(set(targets)) == 2:
        res['AUPRC'] = average_precision_score(targets, preds, average='macro')
        res['brier'] = brier_score_loss(targets, preds)
        res['mean_pred_1'] = preds[targets == 1].mean()
        res['mean_pred_0'] = preds[targets == 0].mean()

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds

    return res
    