from typing import List

import torch
from torch import Tensor

import numpy as np

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoderWithLM
from hw_asr.metric.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class WERMetricWithLM(BaseMetric):
    def __init__(self, text_encoder: CTCCharTextEncoderWithLM, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        log_probs = log_probs.cpu()
        lengths = log_probs_length.cpu()
        for batch, target_text in enumerate(text):
            length = lengths[batch]
            target_text = BaseTextEncoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode_with_lm(log_probs[batch])
            wers.append(calc_wer(target_text, pred_text))
        return np.mean(wers)


class BeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        probs = torch.exp(log_probs).cpu()
        lengths = log_probs_length.cpu()
        for batch, target_text in enumerate(text):
            length = lengths[batch]
            target_text = BaseTextEncoder.normalize_text(target_text)
            hypos = self.text_encoder.ctc_beam_search(probs[batch], probs_length=length)
            pred_text = hypos[0].text
            wers.append(calc_wer(target_text, pred_text))
        return np.mean(wers)
