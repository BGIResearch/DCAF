"""Generate training, validation and testing samples in a stream, whose batch goes through process of LLM embedding,
generating and post-processing
"""
import numpy as np
import torch
import os
import random
import gc

from torch.utils.data import IterableDataset, DataLoader
from scipy.sparse import load_npz

import sys
sys.path.append('../..')


class DatasetStream(IterableDataset):
    """Read expression arrays and perturbation list from disk. Then form a flow of cell batches, to get expression data.
    Assigning all training data to gpu, by using torch.utils.data.Dataset, would use 21 G of memory, which is not memory-efficient.

    Have referred to:
    https://blog.csdn.net/autoliuweijie/article/details/121693112
    """
    def __init__(self, prepro_dir, batch_size, sample_type='train'):
        """
        :param prepro_path: Preprocessed data path that saves expression matrix and perturbed gene list.
        :param batch_size: Batch size for both LLM embedding retrieving and further training, evaluating and testing.
        :param sample_type: To generate training, validation or testing data.
        """
        super().__init__()

        assert sample_type in ['train', 'train_eval', 'val'], "sample_type shall be one of 'train', 'train_eval', and 'val'."

        batch_size = int(batch_size)
        assert batch_size >= 1, "batch_size shall be at least 1."

        # load data and uncompress
        self.ctrl_mtx = load_npz(os.path.join(prepro_dir, '{}_ctrl.npz'.format(sample_type.split('_')[0])))  # cell by gene
        self.case_mtx = load_npz(os.path.join(prepro_dir, '{}_case.npz'.format(sample_type.split('_')[0])))  # cell by gene
        self.gene_ids = np.load(os.path.join(prepro_dir, 'gene_id_li.npy'))

        # remove long tail
        # fixme: move to preprocessing, or delete after result is analyzed
        ty2file = {'train': 'train', 'train_eval': 'train', 'val': 'val', 'test': 'test'}
        self.cell_remain = np.load(os.path.join(prepro_dir, '{}_cell_remain.npy').format(ty2file[sample_type]))  # cell [True, False, ...]
        self.ctrl_mtx = self.ctrl_mtx.toarray()[self.cell_remain == 1, :]
        self.case_mtx = self.case_mtx.toarray()[self.cell_remain == 1, :]
        self.all_mtx = np.concatenate([self.ctrl_mtx, self.case_mtx], axis=0)

        # shuffle cells
        if sample_type in ['train', 'val', 'test']:
            # shuffle train, val, test data, to maintain distribution when only certain amount of cells in the front is
            # involved in training, for easier development
            cell_ind = np.arange(self.all_mtx.shape[0])
            random.seed(5)
            random.shuffle(cell_ind)
            self.all_mtx = self.all_mtx[cell_ind, :]
        else:  # train eval data needs to be in different order compared to train data, to calculate an overall train
            # accuracy using first 200 cells
            cell_ind = np.arange(self.all_mtx.shape[0])
            random.seed(6)
            random.shuffle(cell_ind)
            self.all_mtx = self.all_mtx[cell_ind, :]

        # sample cells for faster evaluation
        if sample_type == 'train_eval':
            self.all_mtx = self.all_mtx[:200, :]
            gc.collect()
        elif sample_type == 'val':
            self.all_mtx = self.all_mtx[:200, :]
        # elif sample_type == 'test':
        #     self.all_mtx = self.all_mtx[:500, :]  # fixme

        self.batch_size = batch_size

    def __iter__(self):
        sample_iterator = self._sample_generator()
        return sample_iterator

    def _sample_generator(self):
        """Slice and get a batch, get embedding from LLM, post-process and get input sample for perturbation model.

        :return sample: dict of genetic cell embeddings for control and case, as well as perturbation matrix.
        """
        # construct single row of gene_ids to batch format
        self.gene_ids_batch = torch.tile(torch.unsqueeze(torch.from_numpy(self.gene_ids), 0), (self.batch_size, 1))
        self.gene_ids_batch = self.gene_ids_batch.to(torch.float32)

        i_slice_start = 0
        i_slice_terminate = self.all_mtx.shape[0]  # the distal end slicing index
        while i_slice_start < i_slice_terminate:
            i_slice_end = min([i_slice_start + self.batch_size, i_slice_terminate])

            # slice and retrieve corresponding cells
            mtx_batch = self.all_mtx[i_slice_start:i_slice_end, :]  # on cpu

            # get embedding of cell batch using LLM, on gpu
            mtx_batch = torch.from_numpy(mtx_batch)
            mtx_batch = mtx_batch.to(torch.float32)

            # sum norm, log trans. same as scGPT perturbation and annotation finetune preprocessing
            mtx_batch_in = torch.log1p(mtx_batch / mtx_batch.sum(axis=1, keepdims=True) * 10000)
            sample = {'exp_in': mtx_batch_in,
                      'exp_out': mtx_batch,
                      'id': self.gene_ids_batch}

            i_slice_start = i_slice_end
            yield sample
        return StopIteration()


def prepare_dataloader(
        dataset: IterableDataset,
        num_workers: int = 0
) -> DataLoader:
    """Prepare dataloader for training, validating and testing.

    当 batch_size 与 batch_sampler 同时设置为 None 时（batch_sampler 默认为 None），自动分批会被关闭。从数据集中获取的样本都会由参数
    collate_fn 设置的函数进行处理。自动分批功能被关闭之后，默认的 collate_fn 仅仅会将 NumPy 数组转换为 PyTorch 中 Tensor 的类型，其他不
    变。

    # https://pytorch.org/docs/stable/data.html

    :param dataset: Instantiated streaming dataset.
    :param num_workers: Number of workers.
    :return: Dataloader.
    """

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=None,
        batch_sampler=None,
        num_workers=num_workers,
        shuffle=False,
    )

    return dataloader



