"""Generate training, validation and testing samples in a stream, whose batch goes through process of LLM embedding
generating and post-processing
"""
import numpy as np
import torch
import os
import random
import gc

from torch.utils.data import IterableDataset, DataLoader
from torch import Tensor
from scipy.sparse import load_npz


# from scGPT_finetune import extract_config, extract_vocab, load_model, get_gene_emb as get_emb
from infer_by_ctrl import extract_config, extract_vocab, load_model, get_gene_emb as get_emb

var_names_path = os.path.join(os.getcwd(), 'infer_by_ctrl/ours/data/var_names.npy')
gene_id_li_path = os.path.join(os.getcwd(), 'infer_by_ctrl/ours/data/gene_id_li.npy')
scgpt_finetuned_model_path = os.path.join(os.getcwd(), 'infer_by_ctrl/ours/data/jepa_scgpt.ckpt')

var_names = np.load(var_names_path)
gene_id_li = np.load(gene_id_li_path)

# load scGPT (fine-tuned)
vocab = extract_vocab()
model_configs = extract_config()
llm = load_model(vocab, model_configs, state='finetune', model_file=scgpt_finetuned_model_path)


def _gaussian_norm_along_emb(mtx: Tensor) -> Tensor:
    """Gaussian normalize long LLM embedding

    To approach the distribution of NLP embeddings used by Transformer, gaussian normalize the matrix on the last
    dimension, for more effective attention solving

    :param mtx:  genetic cell embedding, with shape of (n_cell, n_gene_, n_llm) or (n_gene_, n_llm) for selected genes
    of selected cells
    :return mtx_norm: Gaussian normalized cell embedding, with same shape as mtx
    """

    mtx_norm = (mtx - mtx.mean(axis=-1, keepdims=True)) / \
                     (torch.std(mtx, axis=-1, keepdims=True) + 1e-5)
    return mtx_norm


def _gen_cell_pert_mtx(pert_li):
    """
    generate cell index and perturbed gene index matrix based on a perturbing condition list

    :param pert_li: perturbing condition list e.g. np.array(['1_3', '', '5', ...])
    :return cell_pert_mtx: e.g. np.array([[0, 1], [0, 3], [2, 5], ...])
    """
    cell_pert_mtx = np.array([[i, int(gene)]
                              for i in range(pert_li.shape[0])
                              for gene in pert_li[i].split('_')
                              if gene != ''])
    return cell_pert_mtx


def _gen_pert_mtx(g_cell_mtx_ctrl_norm: Tensor, pert_ind_ctrl: Tensor) -> Tensor:
    """Preprocess perturbation matrix so perturbed genes have the same embedding as output by LLM then normalized, and
    unperturbed genes share a same gaussian distribution with same average and std as perturbed ones.

    :param g_cell_mtx: genetic cell embedding by LLM, with shape of (n_cell, n_gene_, n_llm)
    :param pert_ind_ctrl: perturbed gene index of each cell, with shape of n_perturbations_total * 2, with first column
    storing cell index in respective to control cells, and the second dimension storing single gene index which is
    perturbed.

    :return pert_mtx: perturbation matrix, with shape of (n_cell_ctrl, n_gene, d_llm), storing normalized LLM embedding
    of perturbed genes and gaussian distribution embedding of other genes.
    """

    # initialize a perturbation matrix where each gene of each cell has an embedding of gaussian distribution by default,
    # while the distribution has same average (0) and std (1) value as g_cell_mtx_norm
    torch.manual_seed(5)  # every run outputs the same preprocessing result
    gauss_1d = torch.normal(0, 1, size=(g_cell_mtx_ctrl_norm.shape[2],))  # gaussian distribution in 1d
    gauss_1d = gauss_1d.to(g_cell_mtx_ctrl_norm.device, g_cell_mtx_ctrl_norm.dtype)
    pert_mtx = torch.tile(gauss_1d, (g_cell_mtx_ctrl_norm.shape[0], g_cell_mtx_ctrl_norm.shape[1], 1))

    # change embedding of perturbed gene using indices
    pert_mtx[pert_ind_ctrl[:, 0], pert_ind_ctrl[:, 1], :] = g_cell_mtx_ctrl_norm[pert_ind_ctrl[:, 0], pert_ind_ctrl[:, 1], :]

    # in numpy
    # # initialize a perturbation matrix where each gene of each cell has an embedding of gaussian distribution by default,
    # # while the distribution has same average (0) and std (1) value as g_cell_mtx_norm
    # np.random.seed(5)  # every run outputs the same preprocessing result
    # gauss_1d = np.random.normal(loc=0, scale=1, size=g_cell_mtx_ctrl_norm.shape[2])  # gaussian distribution in 1d
    # pert_mtx = np.tile(gauss_1d, (g_cell_mtx_ctrl_norm.shape[0], g_cell_mtx_ctrl_norm.shape[1], 1))
    #
    # # change embedding of perturbed gene using indices
    # pert_mtx[pert_ind_ctrl[:, 0], pert_ind_ctrl[:, 1], :] = g_cell_mtx_ctrl_norm[pert_ind_ctrl[:, 0], pert_ind_ctrl[:, 1], :]
    return pert_mtx


def _uniform_distri_nonzero_gene(gene_ids_batch: Tensor,
                                 ctrl_mtx_batch: Tensor, case_mtx_batch: Tensor,
                                 ctrl_mtx_batch_norm: Tensor, case_mtx_batch_norm: Tensor,
                                 cell_pert_batch: Tensor):
    """
    let non-zero gene distribute uniformly.

    gene_ids_batch: (cell, gene) gene id list of batch of cells.
    ctrl_mtx_batch: (cell, gene) gene expression of batch of control cells. 0 means no expression. Hence log1p pre-preocessing is accepted
        yet log preprocessing is not.
    case_mtx_batch: (cell, gene) gene expression of batch of case cells. 0 means no expression. Hence log1p pre-preocessing is accepted
    yet log preprocessing is not.
    cell_pert_batch: cell * pert_gene_index, gene corresponds to expression matrix, e.g. [[2, 10], [2, 1], [3, 3]]

    return: gene_ids_batch: gene id list where non-zero genes are uniformly distributed
    return: ctrl_mtx_batch: gene expression of control cells where non-zero genes are uniformly distributed
    return: case_mtx_batch: gene expression of case cells where non-zero genes are uniformly distributed
    return: cell_pert_batch: cell * pert_gene_index, gene corresponds to new ordered gene, where non-zero genes are uniformed distributed
    """

    gene_ids_batch_raw = gene_ids_batch.clone()

    ctrl_mtx_batch_raw = ctrl_mtx_batch.clone()
    case_mtx_batch_raw = case_mtx_batch.clone()

    ctrl_mtx_batch_norm_raw = ctrl_mtx_batch_norm.clone()
    case_mtx_batch_norm_raw = case_mtx_batch_norm.clone()

    nonzero_where = (ctrl_mtx_batch_norm_raw > 0)  # (cell, gene) where non zero genes are
    nonzero_no = nonzero_where.sum(axis=1)  # (cell,) count of non-zero genes of each cell in the batch

    init_ind_batch = torch.tile(torch.arange(ctrl_mtx_batch_raw.shape[1]),
                                (ctrl_mtx_batch_raw.shape[0], 1))  # initial index
    init_ind_batch = init_ind_batch.to(ctrl_mtx_batch_raw.device)
    init_ind_batch_raw = init_ind_batch.clone()
    for cell_ind in range(ctrl_mtx_batch_raw.shape[0]):  # cell index

        distri_ind = torch.linspace(0, ctrl_mtx_batch_raw.shape[1] - 1,
                                    nonzero_no[cell_ind].item()).int()  # (no,) index for uniform distribution
        distri = torch.zeros(ctrl_mtx_batch_raw.shape[1])
        distri = distri.to(ctrl_mtx_batch_raw.device, torch.bool)
        distri[distri_ind] = 1

        gene_ids_batch[cell_ind, distri] = gene_ids_batch_raw[cell_ind, nonzero_where[cell_ind]]
        gene_ids_batch[cell_ind, ~distri] = gene_ids_batch_raw[cell_ind, ~nonzero_where[cell_ind]]

        ctrl_mtx_batch[cell_ind, distri] = ctrl_mtx_batch_raw[cell_ind, nonzero_where[cell_ind]]
        ctrl_mtx_batch[cell_ind, ~distri] = ctrl_mtx_batch_raw[cell_ind, ~nonzero_where[cell_ind]]

        case_mtx_batch[cell_ind, distri] = case_mtx_batch_raw[cell_ind, nonzero_where[cell_ind]]
        case_mtx_batch[cell_ind, ~distri] = case_mtx_batch_raw[cell_ind, ~nonzero_where[cell_ind]]

        ctrl_mtx_batch_norm[cell_ind, distri] = ctrl_mtx_batch_norm_raw[cell_ind, nonzero_where[cell_ind]]
        ctrl_mtx_batch_norm[cell_ind, ~distri] = ctrl_mtx_batch_norm_raw[cell_ind, ~nonzero_where[cell_ind]]

        case_mtx_batch_norm[cell_ind, distri] = case_mtx_batch_norm_raw[cell_ind, nonzero_where[cell_ind]]
        case_mtx_batch_norm[cell_ind, ~distri] = case_mtx_batch_norm_raw[cell_ind, ~nonzero_where[cell_ind]]

        init_ind_batch[cell_ind, distri] = init_ind_batch_raw[cell_ind, nonzero_where[cell_ind]]
        init_ind_batch[cell_ind, ~distri] = init_ind_batch_raw[cell_ind, ~nonzero_where[cell_ind]]

    min_cell_idx = cell_pert_batch[:, 0].min().item()
    cell_pert_batch_within = cell_pert_batch.clone()  # cell index within batch
    cell_pert_batch_within[:, 0] = cell_pert_batch_within[:, 0] - min_cell_idx  # [[0, 9], [0, 1], [1, 3]]

    cell_pert_batch_g_col = torch.Tensor([(init_ind_batch[row_ind] == gene_old_ind).nonzero()[0].item()
                                          for (row_ind, gene_old_ind) in
                                          cell_pert_batch_within])  # gene column, the 2nd column
    cell_pert_batch[:, 1].copy_(cell_pert_batch_g_col)

    return gene_ids_batch, ctrl_mtx_batch, case_mtx_batch, ctrl_mtx_batch_norm, case_mtx_batch_norm, cell_pert_batch


class DatasetStream(IterableDataset):
    """Read expression arrays and perturbation list from disk. Then form a flow of cell batches, to get embedding from
    LLM, and post-process them to further feed into perturbation prediction model. Have referred to:
    https://blog.csdn.net/autoliuweijie/article/details/121693112
    """
    def __init__(self, ctrl_mtx, pert_gene_name, batch_size, llm_device):
        """
        :param prepro_path: Preprocessed data path that saves expression matrix and perturbed gene list.
        :param batch_size: Batch size for both LLM embedding retrieving and further training, evaluating and testing.
        :param sample_type: To generate training, validation or testing data.
        :param llm_device: Device for running LLM, 'cpu' or 'cuda'. Post-processing will happen on the same device.
        :param output_device: Device where output data is stored in, 'cpu' or 'cuda'.
        """
        super().__init__()

        batch_size = int(batch_size)
        assert batch_size >= 1, "batch_size shall be at least 1."

        # extract llm vocab and model configs
        self.vocab = vocab
        self.model_configs = model_configs

        self.ctrl_mtx = ctrl_mtx
        self.pert_li = np.array([str(np.where(var_names == pert_gene_name)[0][0])] * self.ctrl_mtx.shape[0])
        self.gene_ids = gene_id_li

        global llm
        self.llm = llm
        self.llm.to(llm_device)
        self.llm.eval()

        self.llm_device = llm_device
        self.batch_size = batch_size

        # construct single row of gene_ids to batch format. Operate here to save time when iterating.
        self.gene_ids_batch = torch.tile(torch.unsqueeze(torch.from_numpy(self.gene_ids), 0), (self.batch_size, 1))
        self.gene_ids_batch = self.gene_ids_batch.to(self.llm_device, torch.float32)

    def __iter__(self):
        sample_iterator = self._sample_generator()
        return sample_iterator

    def _sample_generator(self):
        """Slice and get a batch, get embedding from LLM, post-process and get input sample for perturbation model.

        :return sample: dict of genetic cell embeddings for control and case, as well as perturbation matrix.
        """

        i_slice_start = 0
        i_slice_terminate = self.ctrl_mtx.shape[0]  # the distal end slicing index
        while i_slice_start < i_slice_terminate:
            i_slice_end = min([i_slice_start + self.batch_size, i_slice_terminate])

            # slice and retrieve corresponding cells, gene_ids_batch does not need to be sliced
            ctrl_mtx_batch, pert_li_batch = (
                self.ctrl_mtx[i_slice_start:i_slice_end, :],
                self.pert_li[i_slice_start:i_slice_end]
            )  # pert_li_batch: e.g. ['10_1', '3', ...]  # on cpu

            # get control and case exp matrix
            ctrl_mtx_batch = torch.from_numpy(ctrl_mtx_batch)
            ctrl_mtx_batch = ctrl_mtx_batch.to(self.llm_device, torch.float32)

            # revise batch size, for last iteration
            gene_ids_batch = self.gene_ids_batch[:ctrl_mtx_batch.shape[0], :].clone()  # gpu

            # get embedding of cell batch using LLM, on gpu
            with torch.no_grad():
                g_cell_mtx_ctrl = get_emb(ctrl_mtx_batch, gene_ids_batch, self.vocab, self.model_configs, model=self.llm, max_length=1200)  # fixme：1200为参数
                torch.cuda.empty_cache()

                g_cell_mtx_ctrl = _gaussian_norm_along_emb(g_cell_mtx_ctrl)  # gpu

            # get perturbation matrix
            cell_pert_batch = _gen_cell_pert_mtx(pert_li_batch)  # cpu # out: cell * pert_gene_index, e.g. [[0, 10], [0, 1], [1, 3]]
            cell_pert_batch = torch.from_numpy(cell_pert_batch)
            cell_pert_batch = cell_pert_batch.to(self.llm_device, torch.int)

            pert_mtx = _gen_pert_mtx(g_cell_mtx_ctrl, cell_pert_batch)  # gpu

            torch.cuda.empty_cache()

            sample = {'ctrl': g_cell_mtx_ctrl,  # gpu
                      'pert': pert_mtx,  # gpu
                      'ctrl_exp': ctrl_mtx_batch,  # gpu
                      'pert_li': pert_li_batch}  # cpu

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


