import json
from pathlib import Path
from random import shuffle
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch.utils.data import DataLoader

from datautils import SeqDataset
from models import BiGRUClassifier
from trainutils import device, kfold_train

# Hyperparameter
batch_size = 128
learning_rate = 1e-3
trial_name = "BiGRU-Drop"
num_speakers = 600

# Define Dataset && DataLoader
# with open("./zscore-full.skl", "rb") as fp:
#     scaler = pickle.load(fp)
scaler = torch.jit.load("./scaler.jit")

data_dir = "./Dataset"
mapping_path = Path(data_dir) / "mapping.json"
mapping = json.load(mapping_path.open())
speaker2id = mapping["speaker2id"]

all_data = []
metadata_path = Path(data_dir) / "metadata.json"
metadata = json.load(metadata_path.open())["speakers"]
for speaker in metadata.keys():
    for utterances in metadata[speaker]:
        all_data.append((utterances["feature_path"], speaker2id[speaker]))
shuffle(all_data)

step = int(len(all_data) * 0.2)
a, b = 0, step
train_datasets, valid_datasets = [], []
for _ in range(5):
    valid_data = all_data[a:b]
    train_data = all_data[:a] + all_data[b:]
    train_datasets.append(SeqDataset(data_dir, train_data, scaler))
    valid_datasets.append(SeqDataset(data_dir, valid_data, scaler))
    a = b
    b += step


def collate(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[PackedSequence, Tensor]:
    features, labels = zip(*batch)
    return pack_sequence(features, enforce_sorted=False), torch.cat(labels)


train_dataloaders = [
    DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate)
    for dataset in train_datasets
]
valid_dataloaders = [
    DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate)
    for dataset in valid_datasets
]

# Define Model
model = BiGRUClassifier(embed_size=40, hidden_size=256, num_layers=3, num_classes=num_speakers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
kfold_train(train_dataloaders, valid_dataloaders, model, criterion, optimizer, 8000, 100, trial_name)
