"""
方法总结：
1. 明确函数的输入，输出
1. 找到最相似的模块进行改写
# 2. 了解模块中类和函数的调用关系，各自的功能
# 3. 注释调不用的代码块， 增加和修改
"""

"""Revised by Tianyi Xia, BGI Research, from scgpt.tasks.cell_emb. 

Code reconstructed for easier understanding and convenient revision. 

In terms of context, original code shows process of getting cell embedding. To get gene embedding instead, gene 
expression and gene index arrays are first segmented to be shorter than maximum sequence length. 
"""

import os
from typing import Optional, Union

import numpy as np
import torch
from anndata import AnnData

from .data_collator import DataCollator

PathLike = Union[str, os.PathLike]


def get_emb(
    count_mtx_batch,
    gene_ids_batch,
    vocab,
    model_configs,
    model=None,
    max_length=1200,
    mask_idx=None
) -> torch.Tensor:
    """
    Get the gene embeddings for a batch of cells.
    """

    def add_cls_token(count_mtx_batch_seg, gene_ids_batch_seg, vocab_cls, pad_val):
        batch_size = count_mtx_batch_seg.shape[0]

        gene_ids_batch_seg = torch.cat((torch.ones((batch_size, 1), device=gene_ids_batch_seg.device)*vocab_cls, gene_ids_batch_seg), dim=1)   # np.insert(gene_ids_batch, 0, vocab_cls)  # (batch, genes+1)
        count_mtx_batch_seg = torch.cat((torch.ones((batch_size, 1), device=count_mtx_batch_seg.device)*pad_val, count_mtx_batch_seg), dim=1)  # np.insert(count_mtx_batch, 0, pad_val)

        return gene_ids_batch_seg, count_mtx_batch_seg

    def get_emb_seg(count_mtx_batch_seg, gene_ids_batch_seg, vocab, model_configs, collator, mask_idx):
        seg_gene_num = count_mtx_batch_seg.shape[1]

        gene_ids_batch_seg, count_mtx_batch_seg = add_cls_token(count_mtx_batch_seg, gene_ids_batch_seg, vocab['<cls>'], model_configs['pad_value'])

        genes_ids_batch_seg = gene_ids_batch_seg.long()
        count_mtx_batch_seg = count_mtx_batch_seg.float()

        dataset_ = [{'genes': genes_ids_batch_seg[idx, :],
                     'expressions': count_mtx_batch_seg[idx, :]}
                    for idx in range(count_mtx_batch_seg.shape[0])]

        dataset_padded = collator(dataset_)  # {'gene': (batch_size, gene_num), 'expr', 'masked_expr'}

        # with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):  # fixme
        with torch.cuda.amp.autocast(enabled=True):
            input_gene_ids = dataset_padded['gene']

            # different from mask setting of scgpt.tasks.cell_emb
            # src_key_padding_mask = input_gene_ids.eq(vocab[model_configs['pad_token']])
            src_key_padding_mask = torch.zeros_like(input_gene_ids, dtype=torch.bool, device=input_gene_ids.device)
            if mask_idx is not None:
                src_key_padding_mask[:, mask_idx+1] = 1

            embeddings = model._encode(
                input_gene_ids,
                dataset_padded['expr'],
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None
            )  # (batch, token, dim)

            gene_embeddings_batch_seg = embeddings[:, 1:seg_gene_num+1, :]  # <cls> was added, hence other tokens move backwards by 1 position

        return gene_embeddings_batch_seg

    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[model_configs["pad_token"]],
        pad_value=model_configs["pad_value"],
        do_mlm=False,  # no need to mask when getting embedding
        do_binning=True,
        max_length=max_length+1,  # leave one extra token for <cls>
        sampling=True,
        keep_first_n_tokens=1,
    )

    gene_embeddings_seg_li = []

    num_gene_per_seg = max_length
    start = 0  # slice position
    end = start + num_gene_per_seg

    if end >= count_mtx_batch.shape[1]:  # go through llm for one loop
        count_mtx_batch_seg = count_mtx_batch[:, start:end]
        gene_ids_batch_seg = gene_ids_batch[:, start:end]

        gene_embeddings_seg = get_emb_seg(count_mtx_batch_seg, gene_ids_batch_seg, vocab, model_configs, collator, mask_idx)
        gene_embeddings_seg_li.append(gene_embeddings_seg)
    else:
        while start < count_mtx_batch.shape[1]:

            count_mtx_batch_seg = count_mtx_batch[:, start:end]
            gene_ids_batch_seg = gene_ids_batch[:, start:end]

            gene_embeddings_seg = get_emb_seg(count_mtx_batch_seg, gene_ids_batch_seg, vocab, model_configs, collator, None)
            gene_embeddings_seg_li.append(gene_embeddings_seg)

            start = end
            end = min([start + num_gene_per_seg, count_mtx_batch.shape[1]])
    return torch.cat(gene_embeddings_seg_li, dim=1)

