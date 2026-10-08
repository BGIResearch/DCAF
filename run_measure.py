import sys
import scipy
import os

from argparse import ArgumentParser
from scipy import sparse

from dcaf import RmSilenceGene, NotRmSilenceGene  # todo: 打包后调用包


def _load_data(args):
    if args.mode != 'user':  # for demo
        ctrl = sparse.load_npz(os.path.join(os.getcwd(), 'data', 'simulate_ctrl.npz')).toarray()
        case = sparse.load_npz(os.path.join(os.getcwd(), 'data', 'simulate_case.npz')).toarray()
        pred = sparse.load_npz(os.path.join(os.getcwd(), 'data', 'simulate_pred.npz')).toarray()
    else:  # for user's own data
        if os.path.exists(args.control) and os.path.exists(args.case) and os.path.exists(args.pred):
            ctrl = sparse.load_npz(args.control).toarray()
            case = sparse.load_npz(args.case).toarray()
            pred = sparse.load_npz(args.pred).toarray()
        else:
            ctrl = sparse.load_npz(os.path.join(os.getcwd(), 'data', args.control)).toarray()
            case = sparse.load_npz(os.path.join(os.getcwd(), 'data', args.case)).toarray()
            pred = sparse.load_npz(os.path.join(os.getcwd(), 'data', args.pred)).toarray()
    return ctrl, case, pred


def _decide_low_thresh(args):
    if args.mode != 'user':
        return 0.1
    else:
        return args.low_thresh  # by default, 0.1 as well


def __main__():
    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='user', help='to use or to show demo')  # to measure user's own data by default, otherwise assign 'demo'

    parser.add_argument('-ctrl', '--control', type=str, help='control data path')
    parser.add_argument('-case', '--case', type=str, help='case data path')
    parser.add_argument('-pred', '--pred', type=str, help='predicted data path')

    parser.add_argument('-thrsh', '--low-thresh', dest='low_thresh', default=0.1, type=float,
                        help='Threshold between 0 inflation and binomial distribution')
    args = parser.parse_args()

    # user need to assign complete path or put data in /data if not using demo that is shown    
    if args.mode != 'user':  # demo mode no need to assert
        pass
    else:
        assert args.control and args.case and args.pred, "Control, case or pred path not assigned"
        assert (os.path.exists(args.control) 
                and os.path.exists(args.case) 
                and os.path.exists(args.pred))\
               or\
               (os.path.exists(os.path.join(os.getcwd(), 'data', args.control))
                and os.path.exists(os.path.join(os.getcwd(), 'data', args.case))
                and os.path.exists(os.path.join(os.getcwd(), 'data', args.pred))), \
               "For control, case and pred file,specify their CORRECT file name under /data, or specify their CORRECT full path"

    low_thresh = _decide_low_thresh(args)

    ctrl, case, pred = _load_data(args)
    assert ctrl.shape == case.shape == pred.shape, "Control, case and predicted expression are not of the same size"

    # measure
    rm_sil_gene = RmSilenceGene(ctrl, case, pred, low_thresh=low_thresh)

    not_rm_sil_gene = NotRmSilenceGene(ctrl, case, pred, low_thresh=low_thresh)
    not_rm_sil_gene.cal_log_fold_change()

    rm_sil_gene_measure = {'mse': rm_sil_gene.mse, 'mae': rm_sil_gene.mae}
    not_rm_sil_gene_measure = {'r2_lfc': not_rm_sil_gene.r2_lfc, 'mse_lfc': not_rm_sil_gene.mse_lfc}

    print('Measurement for remain silenced genes,', rm_sil_gene_measure,
          '\n',
          'Measurement for not remain silenced genes', not_rm_sil_gene_measure)


if __name__ == '__main__':
    __main__()
