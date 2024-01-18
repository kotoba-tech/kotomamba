import torch
import numpy as np
import random


def set_seed(seed: int) -> None:
    # Set the seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
