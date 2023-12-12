import numpy as np
from pathlib import Path
import os
import shutil

from tqdm import tqdm

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent

def move_flacs(data_dir = None):
    if data_dir is None:
        data_dir = ROOT_PATH / "data" / "datasets" / "asv_dataset"

    for split in ['dev', 'eval', 'train']:
        current_dir = data_dir / f"ASVspoof2019_LA_{split}" / 'flac'
        wanted_dir = data_dir / split
        wanted_dir.mkdir(parents=False, exist_ok=True)

        for file_name in current_dir.iterdir():
            shutil.move(file_name, wanted_dir)


def move_protocols(data_dir = None):
    if data_dir is None:
        data_dir = ROOT_PATH / "data" / "datasets" / "asv_dataset"

    for split in ['dev', 'eval', 'train']:
        if split != 'train':
            protocol_name = f"ASVspoof2019.LA.cm.{split}.trl.txt"
        else:
            protocol_name = f"ASVspoof2019.LA.cm.{split}.trn.txt"
        protocol_file = data_dir / "ASVspoof2019_LA_cm_protocols" / protocol_name
        protocol_file.replace(data_dir / f"{split}_protocol.txt")


if __name__ == '__main__':
    print('Moving dataset...')
    move_flacs()
    move_protocols()
    print('Finished moving dataset!')
