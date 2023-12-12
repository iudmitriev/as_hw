import logging
from typing import List

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
import numpy as np


def collate_fn(dataset_items: List[dict]):
    result_batch = {}
    values_to_pad = ['is_bonafide', 'audio']
    for value in values_to_pad:
        result_batch[value] = torch.cat([
            item[value] for item in dataset_items
        ])
        if value == 'audio':
            result_batch[value] = result_batch[value].unsqueeze(dim=1)
    return result_batch
