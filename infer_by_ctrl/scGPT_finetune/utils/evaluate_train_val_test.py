import torch
import numpy as np
import pandas as pd
import os


from .loss_and_acc import mse_gene_exp, norm_mse_gene_exp
from .get_gene_exp_by_model import get_exp


class FinetuneEvaluator():
    def __init__(self):
        self.train_df = pd.DataFrame(columns=['epoch', 'iter', 'mse', 'mse_exp_norm'])
        self.val_df = pd.DataFrame(columns=['epoch', 'iter', 'mse', 'mse_exp_norm'])
        # test accuracy along epochs doesn't need to be recorded.

    def eval(self, epoch, iter, model, vocab, model_configs, batch_loader, type, device):
        loss_li = []
        mse_exp_norm_li = []

        for batch in batch_loader:
            # compute training data accuracy
            batch_exp_in = batch['exp_in']
            batch_exp_out = batch['exp_out']
            batch_id = batch['id']

            batch_exp_in = batch_exp_in.to(device)
            batch_exp_out = batch_exp_out.to(device)
            batch_id = batch_id.to(device)

            # encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, \
            # new_data_raw, data_mask_labels, decoder_position_gene_ids = preprocess(batch_exp, pretrainconfig)
            # batch_exp_pred = model.forward(x=encoder_data,
            #                                padding_label=encoder_data_padding,
            #                                encoder_position_gene_ids=encoder_position_gene_ids,
            #                                encoder_labels=encoder_labels,
            #                                decoder_data=decoder_data,
            #                                mask_gene_name=False,
            #                                mask_labels=None,
            #                                decoder_position_gene_ids=decoder_position_gene_ids,
            #                                decoder_data_padding_labels=decoder_data_padding,
            #                                )

            batch_exp_pred, mask = get_exp(batch_exp_in, batch_id, vocab, model_configs, model=model, mode='val')

            loss = mse_gene_exp(batch_exp_pred, batch_exp_out, thresh_0=False)  # [mask]
            mse_exp_norm = norm_mse_gene_exp(batch_exp_pred, batch_exp_out, thresh_0=True)  # [mask]

            loss_li.append(loss.item())
            mse_exp_norm_li.append(mse_exp_norm.item())

        loss = np.array(loss_li).mean()
        mse_exp_norm = np.array(mse_exp_norm_li).mean()

        if type == 'train_eval':
            self.train_df = pd.concat([self.train_df,
                                       pd.DataFrame([{'epoch': epoch,
                                                      'iter': iter,
                                                      'mse': loss,
                                                      'mse_exp_norm': mse_exp_norm}])],
                                      ignore_index=True)
        if type == 'val':
            self.val_df = pd.concat([self.val_df,
                                     pd.DataFrame([{'epoch': epoch,
                                                    'iter': iter,
                                                    'mse': loss,
                                                    'mse_exp_norm': mse_exp_norm}])],
                                    ignore_index=True)
        # test accuracy along epochs doesn't need to be recorded.
        return loss

    def save_history(self, loss_dir):
        self.train_df.to_csv(os.path.join(loss_dir, 'train_eval.csv'))
        self.val_df.to_csv(os.path.join(loss_dir, 'val_eval.csv'))
