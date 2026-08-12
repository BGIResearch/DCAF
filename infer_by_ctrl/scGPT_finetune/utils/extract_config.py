import json
from pathlib import Path
import os


def extract_config():
    # model_dir = Path("/data/work/DCS/PERTURBATION/scGPT_pretrained/model/model")
    # model_dir = Path('/home/share/huadjyin/home/s_caolei2/xiatianyi/DCS/PERTURBATION/scGPT_pretrained/model/model/')
    model_dir = Path(os.getcwd()) / 'infer_by_ctrl/scGPT_finetune/model/model'

    model_config_file = model_dir / "args.json"

    # Retrieve model parameters from config files
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    return model_configs

