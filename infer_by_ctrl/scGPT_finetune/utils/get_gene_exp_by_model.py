
import os
from typing import Union

import torch

from .data_collator import DataCollator

PathLike = Union[str, os.PathLike]


def get_exp(
    count_mtx_batch,
    gene_ids_batch,
    vocab,
    model_configs,
    model=None,
    max_length=1200,
    mode='train',
) -> torch.Tensor:

    """Get the gene expression predicted for a batch of cells, and the masked positions, by scGPT

    gene_exp_infer: inferred expression of all genes, (batch, gene)
    masked_pos: masked position, with True indicating position being masked, (batch, gene)
    """

    def add_cls_token(count_mtx_batch_seg, gene_ids_batch_seg, vocab_cls, pad_val):
        batch_size = count_mtx_batch_seg.shape[0]

        gene_ids_batch_seg = torch.cat((torch.ones((batch_size, 1), device=gene_ids_batch_seg.device)*vocab_cls, gene_ids_batch_seg), dim=1)   # np.insert(gene_ids_batch, 0, vocab_cls)  # (batch, genes+1)
        count_mtx_batch_seg = torch.cat((torch.ones((batch_size, 1), device=count_mtx_batch_seg.device)*pad_val, count_mtx_batch_seg), dim=1)  # np.insert(count_mtx_batch, 0, pad_val)

        return gene_ids_batch_seg, count_mtx_batch_seg

    def get_exp_seg(count_mtx_batch_seg, gene_ids_batch_seg, vocab, model, model_configs, collator, mode):
        def go_thrg_model(seg_gene_num, dataset_padded, vocab, model_configs, model):
            input_gene_ids = dataset_padded['gene']  # (cell, 1200)
            # todo: Is MLM reason why finetuned scF didn't perform well?
            src_key_padding_mask = input_gene_ids.eq(vocab[model_configs['pad_token']])  # (batch_size, gene_num), bool  # (cell, 1200)
            # src_key_padding_mask = torch.zeros_like(input_gene_ids, dtype=torch.bool, device=input_gene_ids.device)

            # Difference compared to perturbation finetune: 1. use all genes without random sampling;
            # to be consistent with model dtype
            dataset_padded['expr'] = dataset_padded['expr'].to(model.state_dict()[list(model.state_dict().keys())[0]].data.dtype)
            embeddings = model._encode(
                input_gene_ids,
                dataset_padded['expr'],
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None
            )  # (batch, token, dim)
            output_values = model.decoder(embeddings)['pred']  # (cell, 1200)
            # output_dict = model(
            #     input_gene_ids,
            #     dataset_padded['expr'],
            #     input_pert_flags,
            #     src_key_padding_mask=src_key_padding_mask,
            #     CLS=CLS,
            #     CCE=CCE,
            #     MVC=MVC,
            #     ECS=ECS,
            # )

            gene_exp_batch_seg = output_values[:, 1:1+seg_gene_num]
            src_key_padding_mask = src_key_padding_mask[:, 1:1+seg_gene_num]
            return gene_exp_batch_seg, src_key_padding_mask

        seg_gene_num = count_mtx_batch_seg.shape[1]

        gene_ids_batch_seg, count_mtx_batch_seg = add_cls_token(count_mtx_batch_seg, gene_ids_batch_seg, vocab['<cls>'], model_configs['pad_value'])
        genes_ids_batch_seg = gene_ids_batch_seg.long()
        count_mtx_batch_seg = count_mtx_batch_seg.float()

        dataset_ = [{'genes': genes_ids_batch_seg[idx, :],
                     'expressions': count_mtx_batch_seg[idx, :]}
                    for idx in range(count_mtx_batch_seg.shape[0])]

        # pretrain: non-zero gene only, binning | mask attention | mask loss
        # fintune by paper: sum norm, log trans, hvg, binning
        # finetune by us: sum norm, log trans, binning | mask attention | no mask loss   # fixme
        dataset_padded = collator(dataset_)  # {'gene': (batch_size, gene_num), 'expr', 'masked_expr'}

        if mode == 'train':
            gene_exp_batch_seg, src_key_padding_mask = go_thrg_model(seg_gene_num, dataset_padded, vocab, model_configs, model)
        else:
            with torch.no_grad():  # torch.cuda.amp.autocast(enabled=True)  # fixme
                gene_exp_batch_seg, src_key_padding_mask = go_thrg_model(seg_gene_num, dataset_padded, vocab, model_configs, model)

        return gene_exp_batch_seg, src_key_padding_mask

    # results in binned values
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[model_configs["pad_token"]],
        pad_value=model_configs["pad_value"],
        do_mlm=False,  # no need to mask when getting embedding
        do_binning=True,
        max_length=max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )

    gene_exp_seg_li = []
    mask_seg_li = []
    num_gene_per_seg = max_length - 1  # leave one token position for <cls>
    start = 0  # slice position
    end = start + num_gene_per_seg
    while start < count_mtx_batch.shape[1]:

        count_mtx_batch_seg = count_mtx_batch[:, start:end]
        gene_ids_batch_seg = gene_ids_batch[:, start:end]

        # gene_embeddings_seg = get_emb_seg(count_mtx_batch_seg, gene_ids_batch_seg, vocab, model, model_configs, collator)
        gene_exp_batch_seg, mask_batch_seg = get_exp_seg(count_mtx_batch_seg, gene_ids_batch_seg, vocab, model,
                                                         model_configs, collator, mode)

        gene_exp_seg_li.append(gene_exp_batch_seg)  # fixme: 可以反向传播吗?
        mask_seg_li.append(mask_batch_seg)

        start = end
        end = min([start + num_gene_per_seg, count_mtx_batch.shape[1]])

    gene_exp_infer = torch.cat(gene_exp_seg_li, dim=1)
    masked_pos = torch.cat(mask_seg_li, dim=1)
    return gene_exp_infer, masked_pos  # todo: return mask for mask finetuning?

