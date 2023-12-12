import json
import logging
import os
import shutil
from curses.ascii import isascii
from pathlib import Path

import torchaudio
from src.base.base_dataset import BaseDataset
from src.utils import ROOT_PATH
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ASVDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "asv_dataset"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            msg = f"Part {part} of asv dataset not found. Use script to download dataset"
            raise FileNotFoundError(msg)
        
        protocol_file = self._data_dir / f"{part}_protocol.txt"
        with open(str(protocol_file), "r") as f:
            data = f.readlines()
        for i, line in tqdm(enumerate(data)):
            line = line.strip()
            audio_name = line.split(' ')[1]
            audio_type = line.split(' ')[4]
            is_bonafide = audio_type == 'bonafide'

            wav_path = split_dir / f"{audio_name}.flac"
            t_info = torchaudio.info(str(wav_path))
            length = t_info.num_frames / t_info.sample_rate

            index.append(
                {
                    "path": str(wav_path),
                    "is_bonafide": is_bonafide,
                    "audio_len": length,
                }
            )

        return index
