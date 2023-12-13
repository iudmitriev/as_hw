import logging
from typing import List

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
import numpy as np


def collate_fn(dataset_items: List[dict]):
    result_batch = {}
    result_batch['is_bonafide'] = torch.cat([
        torch.LongTensor([item['is_bonafide']]) for item in dataset_items
    ])
    result_batch['audio'] = torch.cat([
        item['audio'] for item in dataset_items
    ], dim=0)
    result_batch['audio'] = result_batch['audio'].unsqueeze(dim=1)
    return result_batch
