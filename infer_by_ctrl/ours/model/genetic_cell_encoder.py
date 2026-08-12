"""Blocks need by .model

Design of perturbation encoder block.
"""

from torch import nn, Tensor


class GeneticCellEncoder(nn.Module):
    """The genetic cell encoder

    Projects last dimension of genetic cell matrix to token dimension of predictor. Use 2 layers of MLP.
    """

    def __init__(self, d_llm: int, n_hiddens: int, d_emb: int, dropout_rate: float):
        """
        :param d_llm: dimension size by LLM
        :param n_hiddens: dimension size of hidden layer
        :param d_emb: dimension size of embedding in predictor
        :param dropout_rate: dropout rate adopted by this block
        """
        super().__init__()

        # self.linear1 = nn.Linear(d_llm, n_hiddens)
        # self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(dropout_rate)  # todo: not sure if this is needed

        # self.linear2 = nn.Linear(n_hiddens, d_emb)

        self.linear1 = nn.Linear(d_llm, d_emb)

    def forward(self, g_cell_mtx: Tensor) -> Tensor:  # fixme test by MLP then by conv
        """
        :param g_cell_mtx: Genetic cell matrix, tensor with shape of (n_batch, n_gene, d_llm), storing LLM embedding of
            all genes.
        :return: output of block with shape of (n_batch, n_gene, d_emb).
        """
        # hid_layer = self.dropout(self.activation(self.linear1(g_cell_mtx)))
        # return self.linear2(hid_layer)
        return self.linear1(g_cell_mtx)   # b, gene, 64  # todo