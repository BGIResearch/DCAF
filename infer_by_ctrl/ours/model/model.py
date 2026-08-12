"""
This module constructs the complete predictor model by blocks specified in .blocks
"""
import torch

from torch import nn, Tensor

from .pert_encoder import PerturbationEncoder
from .genetic_cell_encoder import GeneticCellEncoder
from .pert_reg_decoder import PerturbRegDecoder
from .reconstructor import Reconstructor


class PertSimulator(nn.Module):
    """Perturbation simulator

    Consisted of perturbation encoder with encodes perturbation matrix, genetic cell encoder which encodes genetic cell
    matrix, and perturbation-regulation decoder that simulates perturbation and regulation process.
    """
    def __init__(self, d_llm: int, n_hidden_enc: int, dropout_rate: float):
        """
        :param d_llm: dimension size by LLM.
        :param n_hidden_enc: dimension size of hidden layer of both genetic cell encoder and perturbation encoder.
        :param dropout_rate: dropout rate assigned to all blocks
        """
        super().__init__()

        self.d_llm = d_llm  # embedding dimension by LLM

        self.n_hidden_enc = n_hidden_enc  # hidden layer dimension in both perturbation and genetic cell encoders
        self.d_emb = 128  # output dimension of each gene (token), as well as gene dimension in decoder  #  originally 64  # fixme

        self.n_heads = 8  # number of heads in multi-head attention  # best no. on BERT-BASE is 8, 16 # fixme

        self.dropout_rate = dropout_rate  # dropout rate applied in each block

        self.pert_enc = PerturbationEncoder(self.d_llm, self.n_hidden_enc, self.d_emb, self.dropout_rate)
        self.g_cell_enc = GeneticCellEncoder(self.d_llm, self.n_hidden_enc, self.d_emb, self.dropout_rate)
        self.pert_reg_dec = PerturbRegDecoder(self.d_emb, self.n_heads, self.dropout_rate, self.d_llm)

    def forward(self, pert_mtx: Tensor, g_cell_mtx: Tensor) -> Tensor:
        """
        :param pert_mtx:  A Tensor with shape of (n_batch, n_gene, d_llm), storing LLM embedding of perturbed genes and
            uniform embedding of other genes.
        :param g_cell_mtx: Genetic cell matrix, tensor with shape of (n_batch, n_gene, d_llm), storing LLM embedding of
            all genes.
        :return: genetic cell matrix after perturbation and regulation.
        """
        pert_enc_output = self.pert_enc(pert_mtx)
        g_cell_enc_output = self.g_cell_enc(g_cell_mtx)

        g_cell_pert = self.pert_reg_dec(pert_enc_output, g_cell_enc_output)  # , attn_wei_li
        return g_cell_pert  # , attn_wei_li


class PertSimulatorRecons(nn.Module):
    """Perturbation simulator and a reconstructor that reconstructs predicted genetic cell embedding after perturbation
    to gene expression

    Consisted of a Perturbation simulator that predicts perturbation result in the latent (LLM gene embedding) space, and
    a reconstructor that reconstructs gene expression after perturbation.
    """
    def __init__(self, d_llm: int, n_gene: int, n_hidden_enc: int, dropout_rate: float):
        """
        :param d_llm: dimension size by LLM.
        :param n_hidden_enc: dimension size of hidden layer of both genetic cell encoder and perturbation encoder.
        :param dropout_rate: dropout rate assigned to all blocks
        """
        super().__init__()

        self.d_llm = d_llm
        self.n_hidden_enc = n_hidden_enc
        self.dropout_rate = dropout_rate

        self.pert_simu = PertSimulator(self.d_llm, self.n_hidden_enc, self.dropout_rate)
        self.recons = Reconstructor(self.d_llm, n_gene, 100, self.dropout_rate)

    def forward(self, pert_mtx: Tensor, g_cell_mtx: Tensor):
        """
        :param pert_mtx:  A Tensor with shape of (n_batch, n_gene, d_llm), storing LLM embedding of perturbed genes and
            uniform embedding of other genes.
        :param g_cell_mtx: Genetic cell matrix, tensor with shape of (n_batch, n_gene, d_llm), storing LLM embedding of
            all genes.
        :return: gene expression after perturbation and regulation.
        """
        g_cell_emb_pert = self.pert_simu(pert_mtx, g_cell_mtx)
        return self.recons(g_cell_emb_pert)





