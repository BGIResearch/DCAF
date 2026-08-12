__version__ = "0.2.1"
import logging
import sys

logger = logging.getLogger("scGPT")
# check if logger has been initialized
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from . import model, tokenizer, utils  # , scbank, tasks
# from .data_collator import DataCollator
# from .data_sampler import SubsetsBatchSampler
# from .trainer import (
#     prepare_data,
#     prepare_dataloader,
#     train,
#     define_wandb_metrcis,
#     evaluate,
#     eval_testdata,
#     test,
# )

from .utils.extract_config import extract_config
from .utils.extract_vocab import extract_vocab
from .utils.load_model import load_model
from .utils.finetune import finetune
from .utils.dataloader import DatasetStream, prepare_dataloader
from .utils.get_gene_embedding import get_emb as get_gene_emb
