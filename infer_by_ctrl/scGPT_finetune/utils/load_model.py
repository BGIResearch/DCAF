"""
Change parameter loading code, to adapt to lora_pytorch which changes parameter keys in state dictionary
"""

import torch
import sys
import os

from pathlib import Path
from ..model import TransformerModel


def load_model(vocab, model_configs, state='pretrain', model_file=None):
    """
    model structure has loRA.
    """
    assert state in ['pretrain', 'finetune'], "state parameter should be either \'pretrain\' or \'finetune\'"

    n_bins = 51
    pad_value = -2
    n_input_bins = n_bins

    model_dir = Path(os.getcwd()) / 'infer_by_ctrl/scGPT_finetune/model/model/'

    if state == 'pretrain':
        model_file = model_dir / "best_model.pt"  # cover input parameter
    elif state == 'finetune':
        # model_file = '/home/share/huadjyin/home/s_caolei2/xiatianyi/DCS/PERTURBATION/predictor/model_zoo/scgpt_finetune/llm_finetune_scgpt.ckpt'
        # model_file = '/home/share/huadjyin/home/s_caolei2/xiatianyi/DCS/PERTURBATION/predictor/model_zoo/jepa/replogle1/jepa_scgpt.ckpt'
        pass

    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]

    # load model, has lora
    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        pad_value=pad_value,
        n_input_bins=n_input_bins,
    )

    # try:  # when model_file matches perfectly with model parameter
    #     model.load_state_dict(torch.load(model_file))
    #     print(f"Loading all model params from {model_file}")
    # except:
    #     # only load params that are in the model and match the size
    #     model_dict = model.state_dict()
    #     pretrained_dict = torch.load(model_file)
    #     pretrained_dict = {
    #         k: v
    #         for k, v in pretrained_dict.items()
    #         if k in model_dict and v.shape == model_dict[k].shape
    #     }
    #     for k, v in pretrained_dict.items():
    #         print(f"Loading params {k} with shape {v.shape}")
    #         model_dict.update(pretrained_dict)
    #         model.load_state_dict(model_dict)

    model_state_dict = torch.load(model_file)
    # print('model', model.state_dict().keys())
    # print('state dict', model_state_dict.keys())

    if state == 'pretrain':
        for name, param in model.named_parameters():
            if '.module' not in name and '.lora_module' not in name:  # parameter names not affected by lora
                if name in model_state_dict.keys():
                    if model.state_dict()[name].data.shape == model_state_dict[name].data.shape:  # ensure size is matched
                        model.state_dict()[name].copy_(model_state_dict[name].data)

            elif '.module' in name and '.lora_module' not in name:  # parameter names affected by lora, such as original q and v parameters. need to load
                if model.state_dict()[name].data.shape == model_state_dict[
                                                       name.replace('.module', '').replace('in_proj_weight', 'Wqkv.weight').replace('in_proj_bias', 'Wqkv.bias')
                                                   ].data.shape:  # ensure size is matched
                    model.state_dict()[name].copy_(model_state_dict[
                                                       name.replace('.module', '').replace('in_proj_weight', 'Wqkv.weight').replace('in_proj_bias', 'Wqkv.bias')
                                                   ].data)

            else:  # means '.lora_module' in name:
                pass
    elif state == 'finetune':
        model.load_state_dict(model_state_dict)
    return model


