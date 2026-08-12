"""Blocks need by .model

Design of reconstructor block.
"""

import torch

from torch import nn, Tensor


class Reconstructor(nn.Module):
    """The reconstructor that reconstructs predicted genetic cell embedding to expression
    """

    def __init__(self, d_llm: int, n_gene: int, n_hiddens: int, dropout_rate: float):
        """
        :param d_llm: dimension size by LLM
        :param n_hiddens: dimension size of hidden layer
        :param d_recons: dimension size of reconstructed result
        :param dropout_rate: dropout rate adopted by this block
        """
        super().__init__()

        self.linear1 = nn.Linear(n_gene, 100)

        self.linear2 = nn.Linear(100, n_gene)
        self.activation = nn.LeakyReLU()

        self.linear3 = nn.Linear(d_llm, 100)
        self.linear4 = nn.Linear(100, 1)

    def forward(self, g_cell_mtx: Tensor) -> Tensor:
        """
        :param g_cell_mtx: Genetic cell matrix, tensor with shape of (n_batch, n_gene, d_llm), storing LLM embedding of
            all genes predicted.
        :return: reconstructed expression with shape of (n_batch, n_gene).
        """

        g_cell_mtx_trans = g_cell_mtx.transpose(1, 2)

        with torch.no_grad():
            hid1 = self.linear1(g_cell_mtx_trans)
        # hid1 = self.activation(self.linear1(g_cell_mtx_trans))

        hid2 = self.activation(self.linear2(hid1))
        hid2_trans = hid2.transpose(1, 2)  # (n_cell, 19266, 512)

        hid3 = self.activation(self.linear3(hid2_trans))
        return torch.squeeze(self.linear4(hid3), -1)


# # same as scGPT
# class Reconstructor(nn.Module):
#     def __init__(
#             self,
#             d_llm: int,
#     ):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(d_llm, d_llm),  # todo: 区别不大的话，可以使用predictor维度
#             nn.LeakyReLU(),
#             nn.Linear(d_llm, d_llm),
#             nn.LeakyReLU(),
#             nn.Linear(d_llm, 1),
#         )
#
#     def forward(self, x: Tensor):
#         return self.fc(x).squeeze(-1)


# class Reconstructor(nn.Module):
#     """The reconstructor that reconstructs predicted genetic cell embedding to expression
#     """
#
#     def __init__(self, d_llm: int, n_gene: int, n_hiddens: int, dropout_rate: float):
#         """
#         :param d_llm: dimension size by LLM
#         :param n_hiddens: dimension size of hidden layer
#         :param d_recons: dimension size of reconstructed result
#         :param dropout_rate: dropout rate adopted by this block
#         """
#         super().__init__()
#
#         self.linear1 = nn.Linear(d_llm, 32)
#
#         self.recons_blk_first = TransBlock(32, 4, dropout_rate, True)
#         self.recons_blks = nn.ModuleList()
#         for i in range(2):  # 3 layers all together
#             self.recons_blks.append(TransBlock(32, 4, dropout_rate, True))
#
#         self.linear2 = nn.Linear(32, 1)
#
#     def forward(self, g_cell_mtx: Tensor) -> Tensor:
#         """
#         :param g_cell_mtx: Genetic cell matrix, tensor with shape of (n_batch, n_gene, d_llm), storing LLM embedding of
#             all genes predicted.
#         :return: reconstructed expression with shape of (n_batch, n_gene).
#         """
#
#         hid1 = self.linear1(g_cell_mtx)
#
#         y, _ = self.recons_blk_first(hid1, hid1, hid1, hid1)  # (n_batch, n_gene, d_llm)
#         for layer in self.recons_blks:
#             y, _ = layer(y, y, y, y)
#
#         return torch.squeeze(self.linear2(y), -1)


# def _transpose_qkv(X, num_heads):
#     """transpose shape of tensor for computing multi-head attention
#     param X: input, in shape of (batch_size, num_queries, num_hiddens)
#
#     return: outout, in shape of (batch_size，num_queries，num_heads，num_hiddens/num_heads)
#     """
#     X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
#
#     X = X.permute(0, 2, 1, 3)  # (batch_size, num_heads, num of queries, num_hiddens/num_heads)
#
#     return X.reshape(-1, X.shape[2], X.shape[3])
#
#
# def _transpose_output(X, num_heads):
#     """reverse transpose_qkv"""
#     X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
#     X = X.permute(0, 2, 1, 3)
#     return X.reshape(X.shape[0], X.shape[1], -1)
#
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, key_size, query_size, value_size, num_hiddens,
#                  num_heads, dropout, bias=False, **kwargs):
#         super(MultiHeadAttention, self).__init__(**kwargs)
#         self.num_heads = num_heads
#         self.dropout = dropout
#
#         self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
#         self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
#         self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
#         self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
#
#     def forward(self, queries, keys, values):
#         """
#         param queries: (batch_size, num_queries, num_hiddens)
#         param keys: (batch_size, num_queries, num_hiddens)
#         param values: (batch_size, num_queries, num_hiddens)
#         param valid_lens: (batch_size,) or (batch_size, num_queries)
#
#         return
#         """
#         queries = _transpose_qkv(self.W_q(queries), self.num_heads)
#         keys = _transpose_qkv(self.W_k(keys), self.num_heads)
#         values = _transpose_qkv(self.W_v(values), self.num_heads)
#
#         # if valid_lens is not None:
#         # repeat the first term (scaler or vector) for num_heads times, on axis 0, then repeat the 2nd term, and so on
#         #     valid_lens = torch.repeat_interleave(
#         #         valid_lens, repeats=self.num_heads, dim=0)
#
#         # output shape: (batch_size*num_heads, num_queries, num_hiddens/num_heads)
#         # for using optimized scaled dot product attention, among options including flash-attention-2, output remains exactly the same
#         output = nn.functional.scaled_dot_product_attention(queries, keys, values, dropout_p=self.dropout)
#         # fixme: add attention map calculation when studying method result
#
#         # output_concat shape:(batch_size, num_queries, num_hiddens)
#         output_concat = _transpose_output(output, self.num_heads)
#         return self.W_o(output_concat)
#
#
# class AddNorm(nn.Module):
#     """Dropout, add residual, and layer normalize
#
#     Residual connection of input of multi-head attention onto drop-outed result of multi-head attention output, then
#     apply layer normalization (over the last dimension)
#     """
#     def __init__(self, dropout_rate: float, normalized_shape: int):
#         """
#         :param dropout_rate: dropout rate
#         :param normalized_shape: shared size of last dimension of two expected inputs
#         """
#         super().__init__()
#         self.dropout = nn.Dropout(dropout_rate)
#         self.layer_norm = nn.LayerNorm(normalized_shape)
#
#     def forward(self, X: Tensor, Y: Tensor) -> Tensor:
#         """
#         :param X: input of multi-head attention, with shape of (n_batch, n_gene, d_emb)
#         :param Y: output by multi-head attention, same shape as X
#         :return: output of block with shape of (n_batch, n_gene, d_emb)
#         """
#         res_con = self.dropout(Y) + X  # residual connection result
#         return self.layer_norm(res_con)
#
#
# class PositionWiseFFN(nn.Module):
#     """Position-wise feed forward network, where different tokens share the same projection function
#
#     dense, relu then dense
#     """
#     def __init__(self, d_input: int, d_hidden: int, d_output: int):
#         """
#         :param d_input: input dimension
#         :param d_hidden: hidden layer dimension
#         :param d_output: output layer dimension
#         """
#         super().__init__()
#         self.linear1 = nn.Linear(d_input, d_hidden)
#         self.activation = nn.ReLU()
#         self.linear2 = nn.Linear(d_hidden, d_output)
#
#     def forward(self, X: Tensor) -> Tensor:
#         """
#         :param X: output of multi-head attention, with shape of (n_batch, n_gene, d_emb)
#         :return: output of block, with shape of (n_batch, n_gene, d_output)
#         """
#         hid_layer = self.activation(self.linear1(X))
#         return self.linear2(hid_layer)
#
#
# class TransBlock(nn.Module):
#     """same structure as transformer encoder block
#
#     Multi-head attention, with dropout, residual, and layer normalization; then position-wise FFN, with dropout,
#     residual, and layer normalization
#     """
#     def __init__(self, d_emb: int, n_heads: int, dropout_rate: float, use_bias: bool):
#         """
#         :param d_emb: dimension size of embedding in predictor
#         :param n_heads: number of heads in multi-head attention
#         :param dropout_rate: dropout rate in both attention and residual blocks
#         :param use_bias: use bias in attention or not
#         """
#         assert d_emb % n_heads == 0, "embedding dimension should be devisible by heads number"
#
#         super().__init__()
#
#         # self.attention = nn.MultiheadAttention(d_emb, n_heads, dropout_rate, use_bias, batch_first=True)
#
#         # fixme: custom module for using optimized scaled dot product attention, among options including flash-attention-2
#         self.attention = MultiHeadAttention(d_emb, d_emb, d_emb, d_emb, n_heads, dropout_rate, bias=use_bias)
#
#         self.add_norm1 = AddNorm(dropout_rate, d_emb)
#
#         self.ffn = PositionWiseFFN(d_emb, d_emb, d_emb)  # fixme: increase feed foward dimenson?
#         self.add_norm2 = AddNorm(dropout_rate, d_emb)
#
#     def forward(self, q: Tensor, k: Tensor, v: Tensor, ori_sol: Tensor) -> Tensor:
#         """
#         :param k: key matrix as input of multi-head attention of this block, with shape of (n_batch, n_gene, d_emb)
#         :param q: query matrix as input of multi-head attention of this block, with shape of (n_batch, n_gene, d_emb)
#         :param v: value matrix as input of multi-head attention of this block, with shape of (n_batch, n_gene, d_emb)
#         :param ori_sol: original solution as residual term, being one of key, query and value
#         :return: y: output of block, with shape of (n_batch, n_gene, d_emb)
#                  attn_wei: attention weights averaged across heads, with shape of (n_batch, n_gene, n_gene)
#         """
#         # using nn.MultiheadAttention
#         # attn_output, attn_wei = self.attention(q, k, v)
#
#         # fixme: using custom module
#         attn_output = self.attention(q, k, v)
#         attn_wei = None
#
#         atten_res = self.add_norm1(ori_sol, attn_output)  # layer after attention with residual operation
#         y = self.add_norm2(atten_res, self.ffn(atten_res))
#         return y, attn_wei