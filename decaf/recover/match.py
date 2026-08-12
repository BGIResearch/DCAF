import numpy as np
import ot

from typing import Tuple


def kl_divergence_backend(
        ctrl: np.ndarray,
        case: np.ndarray
) -> np.ndarray:
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.
    """
    X_sum_norm = np.expm1(ctrl)  # sum to median expression of each cell
    Y_sum_norm = np.expm1(case)
    X = X_sum_norm / np.tile(np.expand_dims(X_sum_norm.sum(1), 1), (1, ctrl.shape[1])) + 1e-7  # sum to 1
    Y = Y_sum_norm / np.tile(np.expand_dims(Y_sum_norm.sum(1), 1), (1, case.shape[1])) + 1e-7  # sum to 1
    log_X = np.log(X)
    log_Y = np.log(Y)

    X_log_X = np.einsum('ij,ij->i', X, log_X)
    X_log_X = np.reshape(X_log_X, (1, X_log_X.shape[0]))
    M = X_log_X.T - np.dot(X, log_Y.T)
    return np.array(M)


def match_ctrl(
        exp_mtx_ctrl_cand: np.ndarray,
        exp_mtx_case: np.ndarray,
        reg: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find unperturbed status for each case cell.

    :param exp_mtx_ctrl_cand: Candidate matrix of control cells to be matched for case cells.
    :param exp_mtx_case: Case cell matrix.
    :param reg: Regularization factor for Sinkhorn OT matching. Set as 0.5 in DCAF, unless user intends to change this parameter.
    :return:
        tuple: (matched control expression, index of exp_mtx_ctrl_cand selected as match for case cells)
    """
    M = kl_divergence_backend(exp_mtx_ctrl_cand, exp_mtx_case)
    a = np.ones((exp_mtx_ctrl_cand.shape[0],)) / exp_mtx_ctrl_cand.shape[0]
    b = np.ones((exp_mtx_case.shape[0],)) / exp_mtx_case.shape[0]

    match = ot.sinkhorn(a, b, M, reg, verbose=True)
    ctrl_ind_match = np.argmax(match, axis=0)
    ctrl_match = exp_mtx_ctrl_cand[ctrl_ind_match, :]
    return ctrl_match, ctrl_ind_match
