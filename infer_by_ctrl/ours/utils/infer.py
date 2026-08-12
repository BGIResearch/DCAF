import torch
import numpy as np
import sys
import gc
import os

from ours.model.model import PertSimulatorRecons
from ours.utils.dataloader_scGPT_norm_log import DatasetStream, prepare_dataloader

n_gene = 5025
pert_recons_path = os.path.join(os.getcwd(), 'infer_by_ctrl/ours/data/jepa_recons.ckpt')

model = PertSimulatorRecons(d_llm=512, n_gene=n_gene, n_hidden_enc=64, dropout_rate=0.1)
model_data = torch.load(pert_recons_path, map_location='cpu')
model.load_state_dict(model_data)


def infer(adata, pert_gene_name, batch_size=1, infer_device='cuda', label='case_pred'):
    num_test_cell = adata.shape[0]

    global model
    model.to(infer_device)
    model.eval()

    test_dataset = DatasetStream(adata.X, pert_gene_name, batch_size, infer_device)
    test_loader = prepare_dataloader(test_dataset)

    test_exp_pred_all = np.zeros((0, n_gene), dtype=np.float16)
    
    print('{} perturbed, labeled as \'{}\''.format(pert_gene_name, label))
    with torch.no_grad():
        for iter_, batch in enumerate(test_loader):
            print('\r', end='')
            print('Infer progress: {}%'.format(int(iter_*batch_size/num_test_cell*100)), '▋' * iter_, end='')
            sys.stdout.flush()

            torch.cuda.empty_cache()
            test_pert, test_ctrl = (
                batch['pert'],
                batch['ctrl']
            )

            test_exp_pred = model(test_pert, test_ctrl)[:, :].detach().clone().cpu().numpy()

            test_exp_pred_all = np.concatenate([test_exp_pred_all, test_exp_pred], axis=0)
            gc.collect()
        print('\r', end='')
        print('Infer progress: 100%', '▋' * (iter_+1))

    # add predicted case expression to adata
    adata.layers[label] = test_exp_pred_all
    return adata