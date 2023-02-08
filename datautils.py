import json
import os
from pathlib import Path
from typing import Tuple, List

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    def __init__(self, data_dir: str, data: List[Tuple[str, int]], scaler: nn.Module = None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.data = data
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        feat_path, speaker = self.data[idx]
        # Load preprocessed mel-spectrogram
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        if self.scaler:
            mel = self.scaler(mel)
        return mel.float(), torch.FloatTensor([speaker]).long()


class InferenceDataset(Dataset):
    def __init__(self, data_dir: str, scaler: nn.Module=None) -> None:
        super().__init__()
        testdata_path = Path(data_dir) / "testdata.json"
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata["utterances"]
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        if self.scaler:
            mel = self.scaler(mel)
        return feat_path, torch.FloatTensor(mel)
