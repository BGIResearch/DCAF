import os
import numpy as np

from argparse import ArgumentParser
from scipy import sparse
from dcaf import recover_control_status  # todo: 打包后调用包


def _load_data(args):
    if args.mode != 'user':  # for demo
        ctrl = sparse.load_npz(os.path.join(os.getcwd(), 'data', 'ctrl_tian.npz')).toarray()
        case = sparse.load_npz(os.path.join(os.getcwd(), 'data', 'case_tian.npz')).toarray()
    else:  # for user's own data
        if os.path.exists(args.control) and os.path.exists(args.case):
            ctrl = sparse.load_npz(args.control).toarray()
            case = sparse.load_npz(args.case).toarray()
        else:
            ctrl = sparse.load_npz(os.path.join(os.getcwd(), 'data', args.control)).toarray()
            case = sparse.load_npz(os.path.join(os.getcwd(), 'data', args.case)).toarray()
    return ctrl, case


def __main__():
    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='user', help='to use or to show demo')  # to measure user's own data by default, otherwise assign 'demo'

    parser.add_argument('-ctrl', '--control', type=str, help='control data path')
    parser.add_argument('-case', '--case', type=str, help='case data path')
    args = parser.parse_args()

    # user need to assign complete path or put data in /data if not using demo that is shown    
    if args.mode != 'user':  # demo mode no need to assert
        pass
    else:
        assert args.control and args.case and args.pred, "Control, case or pred path not assigned"
        assert (os.path.exists(args.control) 
                and os.path.exists(args.case))\
               or\
               (os.path.exists(os.path.join(os.getcwd(), 'data', args.control))
                and os.path.exists(os.path.join(os.getcwd(), 'data', args.case))
                )), \
               "For control and case file, specify their CORRECT file name under /data, or specify their CORRECT full path"

    ctrl, case = _load_data(args)
    ctrl_match, ctrl_ind_match = recover_control_status(ctrl, case)
    print('{} out of {} control cells are matched.'.format(np.unique(ctrl_ind_match).shape[0], ctrl.shape[0]))


if __name__ == '__main__':
    __main__()
