"""Loss functions need by model training and evaluation.

MSE loss between predicted gene expression and label gene expression, MSE of top 20 DEG
"""
import torch

from typing import List
from torch import Tensor


def mse_gene_exp(exp_pred: Tensor,
                 exp_true: Tensor,
                 gene_ind: Tensor = None,
                 thresh_0: bool = False) -> float:
    """Calculate MSE loss of expression of gene specified with indices.

    :param exp_pred: gene expression after perturbation, predicted by algorithm, in shape of (n_cell, n_gene),
    :param exp_true: ground truth gene expression after perturbations, in shape of (n_cell, n_gene)
    :param gene_ind: gene specified with indices on which MSE is to be calculated.
    :param thresh_0: to replace negative predicted value as 0 or not
    :return:
    """
    if thresh_0:
        exp_pred[exp_pred < 0] = 0

    if gene_ind is None:
        mse = ((exp_pred - exp_true) ** 2).mean()
    else:
        mse = ((exp_pred[:, gene_ind] - exp_true[:, gene_ind]) ** 2).mean()
    return mse


def norm_mse_gene_exp(exp_pred: Tensor,
                      exp_true: Tensor,
                      gene_ind: Tensor = None,
                      thresh_0: bool = True) -> float:
    """Calculate MSE loss of normalized expression of gene specified with indices.

    :param exp_pred: gene expression after perturbation, predicted by algorithm, in shape of (n_cell, n_gene),
    :param exp_true: ground truth gene expression after perturbations, in shape of (n_cell, n_gene)
    :param gene_ind: gene specified with indices on which MSE is to be calculated.
    :param thresh_0: to replace negative predicted value as 0 or not

    :return:
    """

    if thresh_0:
        exp_pred[exp_pred < 0] = 0

    exp_pred_norm = torch.log1p(exp_pred / exp_pred.sum(axis=1, keepdims=True) * 10000)
    exp_true_norm = torch.log1p(exp_true / exp_pred.sum(axis=1, keepdims=True) * 10000)

    if gene_ind is None:
        sqr_ = (exp_pred_norm - exp_true_norm) ** 2
        if torch.isnan(sqr_).sum() >= 1:
            # fixme: remove print when nan problem is solved
            print('raw pred {} causing nan'.format(exp_pred[torch.isnan(sqr_)]))
        mse = sqr_.mean()
    else:
        sqr_ = (exp_pred_norm[:, gene_ind] - exp_true_norm[:, gene_ind]) ** 2
        if torch.isnan(sqr_).sum() >= 1:
            # fixme: remove print when nan problem is solved
            print('raw pred {} causing nan'.format(exp_pred[torch.isnan(sqr_)]))
        mse = sqr_.mean()
    return mse


