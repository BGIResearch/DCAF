import numpy as np

from typing import Tuple


class RmSilenceGene:
    def __init__(
        self,
        ctrl: np.ndarray,
        case_true: np.ndarray,
        case_pred: np.ndarray,
        low_thresh: float
    ):
        self.low_thresh = low_thresh
        self.ctrl_rm, self.case_rm, self.pred_rm = get_remain_sil_gene(ctrl, case_true, case_pred, low_thresh)

    @property
    def mse(self) -> float:
        """
        calculate mse score of expression
        """
        return ((self.pred_rm - self.case_rm) ** 2).mean()

    @property
    def mae(self) -> float:
        """
        calculate mae score of expression
        """
        return (np.abs(self.pred_rm - self.case_rm)).mean()


def get_remain_sil_gene(
        ctrl: np.ndarray,
        case_true: np.ndarray,
        case_pred: np.ndarray,
        low_thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get flattened matrices of cell-gene that remain silenced by ground truth
    :param ctrl: control matrix
    :param case_true: true case matrix
    :param case_pred: predicted case matrix
    :param low_thresh: threshold expression that distinguish zero inflation and negative binomial distribution
    :return: Tuple(flattened control remain silenced, flattened case remain silenced, flattened predicted remain silenced)
    """
    msk_rm_sil = (ctrl < low_thresh) & (case_true < low_thresh)
    return ctrl[msk_rm_sil], case_true[msk_rm_sil], case_pred[msk_rm_sil]

