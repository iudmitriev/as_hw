import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import hydra
import logging

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.object_loading import get_dataloaders

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "checkpoints" / "checkpoint.pth"
DEFAULT_INPUT_PATH = ROOT_PATH / "test_audio"
DEFAULT_RESULTS_PATH = ROOT_PATH / "results.json"


@hydra.main(version_base=None, config_path="src", config_name="config")
def main(config):

    checkpoint_path, in_dir, out_file = parse_args()

    logger = logging.getLogger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = hydra.utils.instantiate(config["arch"])

    logger.info(f"Loading checkpoint: {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    
    in_dir = Path(in_dir)
    test_audios = {}
    for audio_path in in_dir.iterdir():
        audio_tensor, sr = torchaudio.load(audio_path)
        #assert sr == config["preprocessing"]["sr"]
        test_audios[audio_path] = audio_tensor
    
    results = []
    for audio_path, audio in test_audios.items():
        audio = audio.to(device)
        audio = audio.unsqueeze(dim=0)
        predicted_results = model(audio)["prediction"].cpu()
        predicted_results = predicted_results[0]
        print(f'For audio {audio_path.name}')
        print(f'Predicted fake probability = {predicted_results[0].item()}')
        print(f'Predicted bonafide probability = {predicted_results[1].item()}')

        audio_results = {}
        audio_results['audio_path'] = str(audio_path.name)
        audio_results['is bonafide'] = predicted_results[1].item() > predicted_results[0].item()
        audio_results['bonafide probability'] = predicted_results[1].item()
        results.append(audio_results)
    
    with Path(out_file).open("w") as f:
        json.dump(results, f, indent=2)



def parse_args():
    args = argparse.ArgumentParser(description="Pytorch model test")
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH),
        type=str,
        help="path to latest checkpoint (default: checkpoints/checkpoint.pth)",
    )
    args.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_RESULTS_PATH),
        type=str,
        help="File to write results (default: results.json)",
    )
    args.add_argument(
        "-t",
        "--test",
        default=str(DEFAULT_INPUT_PATH),
        type=str,
        help="Path to directory, containing audio to test (default: test_audio/)",
    )
    args = args.parse_args()
    return args.resume, args.test, args.output


if __name__ == "__main__":
    main()
