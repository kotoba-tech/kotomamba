from typing import Type
from llama_recipes.configs.training import train_config
import torch
import numpy as np
import random


def set_seed(train_config: Type[train_config]) -> None:
    # Set the seeds for reproducibility
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
