# todo: 取消decoder最后一层的normalization, 防止LLM中的decoder输入不是normalized
"""Blocks needed by .model

Design of perturbation and regulation decoder block.
"""
import torch

from torch import nn, Tensor


def _transpose_qkv(X, num_heads):
    """transpose shape of tensor for computing multi-head attention
    param X: input, in shape of (batch_size, num_queries, num_hiddens)

    return: outout, in shape of (batch_size，num_queries，num_heads，num_hiddens/num_heads)
    """
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    X = X.permute(0, 2, 1, 3)  # (batch_size, num_heads, num of queries, num_hiddens/num_heads)

    return X.reshape(-1, X.shape[2], X.shape[3])


def _transpose_output(X, num_heads):
    """reverse transpose_qkv"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.dropout = dropout

        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        """
        param queries: (batch_size, num_queries, num_hiddens)
        param keys: (batch_size, num_queries, num_hiddens)
        param values: (batch_size, num_queries, num_hiddens)
        param valid_lens: (batch_size,) or (batch_size, num_queries)

        return
        """
        queries = _transpose_qkv(self.W_q(queries), self.num_heads)
        keys = _transpose_qkv(self.W_k(keys), self.num_heads)
        values = _transpose_qkv(self.W_v(values), self.num_heads)

        # if valid_lens is not None:
        # repeat the first term (scaler or vector) for num_heads times, on axis 0, then repeat the 2nd term, and so on
        #     valid_lens = torch.repeat_interleave(
        #         valid_lens, repeats=self.num_heads, dim=0)

        # output shape: (batch_size*num_heads, num_queries, num_hiddens/num_heads)
        # for using optimized scaled dot product attention, among options including flash-attention-2, output remains exactly the same
        output = nn.functional.scaled_dot_product_attention(queries, keys, values, dropout_p=self.dropout)
        # fixme: add attention map calculation when studying method result

        # output_concat shape:(batch_size, num_queries, num_hiddens)
        output_concat = _transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class AddNorm(nn.Module):
    """Dropout, add residual, and layer normalize

    Residual connection of input of multi-head attention onto drop-outed result of multi-head attention output, then
    apply layer normalization (over the last dimension)
    """
    def __init__(self, dropout_rate: float, normalized_shape: int):
        """
        :param dropout_rate: dropout rate
        :param normalized_shape: shared size of last dimension of two expected inputs
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        :param X: input of multi-head attention, with shape of (n_batch, n_gene, d_emb)
        :param Y: output by multi-head attention, same shape as X
        :return: output of block with shape of (n_batch, n_gene, d_emb)
        """
        res_con = self.dropout(Y) + X  # residual connection result
        return self.layer_norm(res_con)


class PositionWiseFFN(nn.Module):
    """Position-wise feed forward network, where different tokens share the same projection function

    dense, relu then dense
    """
    def __init__(self, d_input: int, d_hidden: int, d_output: int):
        """
        :param d_input: input dimension
        :param d_hidden: hidden layer dimension
        :param d_output: output layer dimension
        """
        super().__init__()
        self.linear1 = nn.Linear(d_input, d_hidden)
        self.activation = nn.LeakyReLU()
        self.linear2 = nn.Linear(d_hidden, d_output)

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: output of multi-head attention, with shape of (n_batch, n_gene, d_emb)
        :return: output of block, with shape of (n_batch, n_gene, d_output)
        """
        hid_layer = self.activation(self.linear1(X))
        return self.linear2(hid_layer)


class PerturbRegDecoderBlock(nn.Module):
    """A single decoder block that simulates either perturbation or gene regulations, with same structure as transformer
    encoder block

    Multi-head attention, with dropout, residual, and layer normalization; then position-wise FFN, with dropout,
    residual, and layer normalization
    """
    def __init__(self, d_emb: int, n_heads: int, dropout_rate: float, use_bias: bool):
        """
        :param d_emb: dimension size of embedding in predictor
        :param n_heads: number of heads in multi-head attention
        :param dropout_rate: dropout rate in both attention and residual blocks
        :param use_bias: use bias in attention or not
        """
        assert d_emb % n_heads == 0, "embedding dimension should be devisible by heads number"

        super().__init__()

        # self.attention = nn.MultiheadAttention(d_emb, n_heads, dropout_rate, use_bias, batch_first=True)

        # fixme: custom module for using optimized scaled dot product attention, among options including flash-attention-2
        self.attention = MultiHeadAttention(d_emb, d_emb, d_emb, d_emb, n_heads, dropout_rate, bias=use_bias)

        self.add_norm1 = AddNorm(dropout_rate, d_emb)

        self.ffn = PositionWiseFFN(d_emb, d_emb, d_emb)  # fixme: increase feed foward dimenson? no difference
        self.add_norm2 = AddNorm(dropout_rate, d_emb)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, ori_sol: Tensor) -> Tensor:
        """
        :param k: key matrix as input of multi-head attention of this block, with shape of (n_batch, n_gene, d_emb)
        :param q: query matrix as input of multi-head attention of this block, with shape of (n_batch, n_gene, d_emb)
        :param v: value matrix as input of multi-head attention of this block, with shape of (n_batch, n_gene, d_emb)
        :param ori_sol: original solution as residual term, being one of key, query and value
        :return: y: output of block, with shape of (n_batch, n_gene, d_emb)
                 attn_wei: attention weights averaged across heads, with shape of (n_batch, n_gene, n_gene)
        """
        # using nn.MultiheadAttention
        # attn_output, attn_wei = self.attention(q, k, v)

        # fixme: using custom module
        attn_output = self.attention(q, k, v)
        attn_wei = None

        atten_res = self.add_norm1(ori_sol, attn_output)  # layer after attention with residual operation
        y = self.add_norm2(atten_res, self.ffn(atten_res))
        return y, attn_wei


class PerturbRegDecoder(nn.Module):
    """Decoder that simulates perturbation and regulation process.

    The first block uses cross attention models perturbation, then coming blocks uses self attention to model gene
    regulation and other pathway regulations such as metabolic pathways. The last block projects token dimension to gene
    embedding dimension specified by LLM.
    """
    def __init__(self, d_emb: int, n_heads: int, dropout_rate: float, d_llm: int):
        """
        :param d_emb: embedding dimension in the decoder
        :param n_heads: number of heads in multi-head attention
        :param dropout_rate: dropout rate assigned to perturbation and regulation decoder
        :param d_llm: LLM embedding dimension
        """
        super().__init__()

        self.d_emb = d_emb  # decoder dimension of each gene (token)
        self.n_heads = n_heads  # number of heads in multi-head attention
        self.n_blk = 3  # number of decoder blocks in total, including both a perturbation block using cross-attention \
        # and regulation blocks using self-attention
        self.use_bias = True  # use bias or not in multi-head attention  # False in all previous experiments

        self.pert_blk = PerturbRegDecoderBlock(self.d_emb, self.n_heads, dropout_rate, self.use_bias)  # perturbation
        # block using cross-attention

        self.reg_blks = nn.ModuleList()  # list of regulation blocks using self-attention
        for i in range(self.n_blk-1):
            self.reg_blks.append(PerturbRegDecoderBlock(self.d_emb, self.n_heads, dropout_rate, self.use_bias))

        self.linear1 = nn.Linear(self.d_emb, d_llm)

    def forward(self, pert_enc: Tensor, g_cell_enc: Tensor) -> Tensor:
        """
        :param pert_enc: output of perturbation encoder, with shape of (n_batch, n_gene, d_emb)
        :param g_cell_enc: output of genetic cell encoder, with shape of (n_batch, n_gene, d_emb)
        :return: output of decoder, with shape of (n_batch, n_gene, d_emb)
                attn_wei_li: list of attention weights averaged across heads, with length equals to number of blocks and
                each weight matrix having shape of (n_batch, n_gene, n_gene)
        """
        # attn_wei_li = []  # for storing attention weights

        # y, attn_wei_crs = self.pert_blk(g_cell_enc, pert_enc, g_cell_enc, g_cell_enc)  # layer that simulates perturbation # todo: may need to change q, k, v：
        y, attn_wei_crs = self.pert_blk(g_cell_enc, g_cell_enc, pert_enc, g_cell_enc)

        # attn_wei_li.append(attn_wei_crs)  # attention weight of cross attention

        for layer in self.reg_blks:  # layers that simulates regulation process
            # todo: may need to change it to cross attention to for each layer to simulate effect by perturbation, similarly to translation by transformer decoder:
            #  解释性变差：高层blk依旧需要原始perturbation embedding的扰动吗？
            y, attn_wei_self = layer(y, y, y, y)
            # attn_wei_li.append(attn_wei_self)  # attention weight of self attention

        return self.linear1(y)  # , attn_wei_li


