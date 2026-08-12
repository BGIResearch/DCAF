import sys
import os

from pathlib import Path

from scGPT_finetune.tokenizer.gene_tokenizer import GeneVocab


def extract_vocab():
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    # Specify model path; here we load the pre-trained scGPT model
    model_dir = Path(os.getcwd()) / 'infer_by_ctrl/scGPT_finetune/model/model/'

    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])
    return vocab

