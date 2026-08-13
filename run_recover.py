import os
import sys
sys.path.append('/')  # fixme
import numpy as np

from scipy.sparse import load_npz

from dcaf import recover_control_status  # todo: 打包后调用包


def  __main__():
    # todo: add using user's own data
    data_dir = '/data'
    ctrl_norman = load_npz(os.path.join(os.getcwd(), data_dir, 'ctrl_tian.npz'))  # todo: 测试后在__init__中定义数据路径
    case_norman = load_npz(os.path.join(os.getcwd(), data_dir, 'case_tian.npz'))
    ctrl_match_norman, ctrl_ind_match_norman = recover_control_status(ctrl_norman.toarray(), case_norman.toarray())
    print('{} out of {} control cells are matched.'.format(np.unique(ctrl_ind_match_norman).shape[0], ctrl_norman.shape[0]))


if __name__ == '__main__':
    __main__()