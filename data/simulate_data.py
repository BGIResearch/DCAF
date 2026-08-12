"""This module provides the function to simulate control, case and model predicting data to demonstrate accuracy measurement. """

import numpy as np

from scipy.stats import nbinom


class SimulateData:
    def __init__(self,
                 ngene: int,
                 ncell: int,
                 p_0: float = 0.83,  # 0 expressed genes proportion # 75% to remain silenced, 8% to be activated
                 n_nbinom: int = 2,
                 p_nbinom: float = 0.1,
                 ratio_acti_sil: float = 0.08
                 ):
        """
        :param ngene: no. of gene
        :param ncell: no. of cell
        :param p_0: ratio of element not expressed across expression matrix
        :param n_nbinom: n value for negative binomial distribution
        :param p_nbinom: p value for negative binomial distribution
        param ratio_acti_sil: ratio of activated genes and (the same) ratio of silenced genes
        """
        self.size = (ncell, ngene)
        self.p_0 = p_0
        self.n_nbinom = n_nbinom
        self.p_nbinom = p_nbinom
        self.ratio_acti_sil = ratio_acti_sil
        self.error_type = None
        self.error_scale = None
        self.mask_infl = None

        self.simulated_control = None
        self.simulated_case = None
        self.simulated_predicted = None

    def simulate_control(self):
        # decide which gene is not expressed
        bernoulli = np.random.binomial(1, self.p_0, self.size)

        # Negative binomial distribution
        neg_binomial = nbinom.rvs(self.n_nbinom, self.p_nbinom, size=self.size)
        neg_binomial = neg_binomial + 10  # actual non_zero gene expression starts from 0, but has usually over 2e4 genes measured

        # normalize total, to simulate labels for perturbation
        exp = np.where(bernoulli == 1, 0, neg_binomial)
        # median_total = np.median(exp.sum(1))
        # exp = exp / exp.sum(1, keepdims=True) * median_total
        self.simulated_control = exp / exp.sum(1, keepdims=True) * 1000

    def simulate_case(self):
        """
        (For different genes of different cells) Permutate and exchange gene expression for genes silenced and activated.
        Permutate expression for genes remain expressed. Keep same values for genes remain silenced.

        For all cell-genes, according to our experiment, 75% remain silenced, 8% silenced, 8% activated, 9% remain expressed.
        """

        no_acti_sil = int(self.size[0] * self.size[1] * self.ratio_acti_sil)

        # specify index of cell-genes, for different consititutions
        # genes activated
        i_0, j_0 = np.where(self.simulated_control == 0)  # ratio: 83%
        index_acti = np.random.permutation(np.arange(i_0.shape[0]))[:no_acti_sil]  # index of array i_0 and j_0
        i_acti = i_0[index_acti]
        j_acti = j_0[index_acti]

        # genes remain silenced
        index_remain_un = np.setdiff1d(np.arange(i_0.shape[0]), index_acti)
        i_remain_un = i_0[index_remain_un]
        j_remain_un = j_0[index_remain_un]

        # genes silenced
        i_1, j_1 = np.where(self.simulated_control > 0)  # ratio: 17%
        index_sil = np.random.permutation(np.arange(i_1.shape[0]))[:no_acti_sil]  # same length as index_acti
        i_sil = i_1[index_sil]
        j_sil = j_1[index_sil]

        # genes remain expressed
        index_remain_exp = np.setdiff1d(np.arange(i_1.shape[0]), index_sil)
        i_remain_exp = i_1[index_remain_exp]
        j_remain_exp = j_1[index_remain_exp]

        # permutate and exchange values for silenced and activated
        inner_i_sil = np.random.permutation(np.arange(i_sil.shape[0]))

        # permutate values for genes remain expressed
        inner_i_remain_exp = np.random.permutation(np.arange(i_remain_exp.shape[0]))

        case_ = np.zeros_like(self.simulated_control)
        case_[i_acti, j_acti] = self.simulated_control[i_sil[inner_i_sil], j_sil[inner_i_sil]]  # activated
        case_[i_sil, j_sil] = 0  # silenced
        case_[i_remain_exp, j_remain_exp] = self.simulated_control[
            i_remain_exp[inner_i_remain_exp], j_remain_exp[inner_i_remain_exp]]  # remain-expressed
        case_[i_remain_un, j_remain_un] = 0  # remain silenced

        self.simulated_case = case_

    def simulate_predicted(self,
                           error_type: str = 'all',
                           scale: float = 0.05,
                           mask_infl: np.ndarray = None):
        """
        two types of error to be injested: 1. uniform Gaussian noise applied on each cell-gene
        2.uniform Gaussian noise applied onto zero-inflated part only
        """

        assert error_type in ['all', 'part'], "error_type value shall be either 'all' or 'part'"
        assert not (
                error_type == 'zero' and mask_infl is None), "if error_type is zero then a mask has to be specified"

        self.error_type = error_type
        self.error_type = error_type
        self.error_scale = scale

        # randomly add error everywhere
        if self.error_type == 'all':
            case_pred = (
                    self.simulated_case
                    + np.random.normal(
                        loc=0,
                        scale=self.error_scale,
                        size=self.size)
                         )

        # partially add error
        else:
            self.mask_infl = mask_infl
            case_pred = np.zeros(self.size)
            case_pred[self.mask_infl] = (
                    self.case_true[self.mask_infl]
                    + np.random.normal(
                        loc=0,
                        scale=scale,
                        size=self.size)[self.mask_infl]
            )
            case_pred[~self.mask_infl] = (
                    self.case_true[~self.mask_infl]
                    + np.random.normal(
                        loc=0,
                        scale=scale,
                        size=self.size)[~self.mask_infl]
            )
        self.simulated_predicted = case_pred




