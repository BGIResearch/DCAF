import numpy as np

from typing import Tuple, Dict
from sklearn.metrics import r2_score


class NotRmSilenceGene:
    def __init__(
        self,
        ctrl: np.ndarray,
        case_true: np.ndarray,
        case_pred: np.ndarray,
        low_thresh: float
    ):
        self.ctrl = ctrl
        self.case_true = case_true
        self.case_pred = case_pred
        self.low_thresh = low_thresh

        self.ctrl_not_rm, self.case_not_rm, self.pred_not_rm = get_not_remain_sil_gene(self.ctrl,
                                                                                       self.case_true,
                                                                                       self.case_pred,
                                                                                       self.low_thresh)
        self.log_fold_change = None

    def cal_log_fold_change(self):
        """
        calculate log fold change.
        """
        fch_true = (
                np.log2(np.expm1(self.case_not_rm) + np.expm1(self.low_thresh))
                - np.log2(np.expm1(self.ctrl_not_rm) + np.expm1(self.low_thresh))
        )
        fch_pred = (
                np.log2(np.expm1(self.pred_not_rm) + np.expm1(self.low_thresh))
                - np.log2(np.expm1(self.ctrl_not_rm) + np.expm1(self.low_thresh))
        )

        # post-processing: in case of nan
        msk_valid = (~np.isnan(fch_true)) & (~np.isnan(fch_pred))
        self.log_fold_change = {'true': fch_true[msk_valid], 'pred': fch_pred[msk_valid]}

    @property
    def r2_lfc(self):
        """
        calculate score of r2 of log fold change
        """
        assert self.log_fold_change is not None, "cal_log_fold_change has not been called"
        return r2_score(self.log_fold_change['true'], self.log_fold_change['pred'])

    @property
    def mse_lfc(self):
        """
        calculate mse of log fold change
        """
        assert self.log_fold_change is not None, "cal_log_fold_change has not been called"
        return np.mean((self.log_fold_change['true'] - self.log_fold_change['pred']) ** 2)


def get_not_remain_sil_gene(
        ctrl: np.ndarray,
        case_true: np.ndarray,
        case_pred: np.ndarray,
        low_thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get flattened matrices of cell-gene that does not remain silenced by ground truth
    :param ctrl: control matrix
    :param case_true: true case matrix
    :param case_pred: predicted case matrix
    :param low_thresh: threshold expression that distinguish zero inflation and negative binomial distribution
    :return: Tuple(flattened control not remain silenced, flattened case not remain silenced, flattened predicted not remain silenced)
    """
    msk_not_rm_sil = (ctrl >= low_thresh) | (case_true >= low_thresh)
    return ctrl[msk_not_rm_sil], case_true[msk_not_rm_sil], case_pred[msk_not_rm_sil]

