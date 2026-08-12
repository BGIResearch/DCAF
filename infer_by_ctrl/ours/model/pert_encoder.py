"""Blocks needed by .model

ADesign of perturbation encoder block.
"""

from torch import nn, Tensor


class PerturbationEncoder(nn.Module):
    """The perturbation encoder.

    Projects last dimension of gene matrix with peturbed gene embedding information to token dimension of predictor. Use
    2 layers of MLP.
    """
    def __init__(self, d_llm: int, n_hidden: int, d_emb: int, dropout_rate: float):
        """
        :param d_llm: dimension size by LLM
        :param n_hidden: dimension size of hidden layer
        :param d_emb: dimension size of embedding in predictor
        :param dropout_rate: dropout rate adopted by this block
        """
        super().__init__()

        # self.linear1 = nn.Linear(d_llm, n_hidden)
        # self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(dropout_rate)  # todo: not sure if this is needed

        # self.linear2 = nn.Linear(n_hidden, d_emb)

        self.linear1 = nn.Linear(d_llm, d_emb)

    def forward(self, pert_mtx: Tensor) -> Tensor:  # fixme test by MLP then by conv
        """
        :param pert_mtx: A Tensor with shape of (n_batch, n_gene, d_llm), storing LLM embedding of perturbed genes and
            uniform embedding of other genes.
        :return: output of block with shape of (n_batch, n_gene, d_emb).
        """
        # hid_layer = self.dropout(self.activation(self.linear1(pert_mtx)))
        # return self.linear2(hid_layer)
        return self.linear1(pert_mtx)  # b, gene, 64