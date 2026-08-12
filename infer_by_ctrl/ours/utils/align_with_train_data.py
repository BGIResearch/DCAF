import scanpy as sc
import numpy as np
import anndata

import os

train_var_names = np.load(os.path.join(os.getcwd(), 'infer_by_ctrl/ours/data/var_names.npy'))


def preprocess(adata):
    """QC, normalize, then log1p"""
    # qc
    sc.pp.filter_cells(adata, min_counts=200, inplace=True)

    # preprocessing
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


def align_with_train_data(adata):
    """reorder gene according to trained data (remove untrained genes, set expression of un-measured gene to 0, reorder)"""
    # remove untrained genes
    trained_genes = np.array([gene for gene in adata.var_names if gene in train_var_names])
    adata = adata[:, trained_genes]
    
    # insert 0 to un-measured genes
    adata_gene = adata.var_names
    un_mea_gene = np.array([gene for gene in train_var_names if not gene in adata_gene])
    if un_mea_gene.shape[0] == 0:
        pass
    else:
        print('Unmeasured gene in dataset:',
              '\n', un_mea_gene)
        print('expression of these genes are set as 0.')

        adata = anndata.AnnData(
            np.concatenate([adata.X, np.zeros((adata.shape[0], un_mea_gene.shape[0]))], axis=1))
        adata.var_names = np.concatenate([adata_gene, un_mea_gene], axis=0)
        adata_gene = adata.var_names  # renew

    # reorder
    ind_ = np.array([np.where(adata_gene == gene)[0][0] for gene in train_var_names])
    adata_aligned = adata[:, ind_]
    return adata_aligned
