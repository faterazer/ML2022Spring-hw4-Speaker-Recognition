import csv
import json
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datautils import InferenceDataset
from models import BiGRUClassifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info]: Use {device} now!")

data_dir = "./Dataset"
mapping_path = Path(data_dir) / "mapping.json"
mapping = json.load(mapping_path.open())

# with open("./zscore-full.skl", "rb") as fp:
scaler = torch.jit.load("./scaler.jit")
dataset = InferenceDataset(data_dir, scaler)


def inference_collate_batch(batch: List[Tuple[str, Tensor]]) -> Tuple[str, PackedSequence]:
    features_paths, mel_features = zip(*batch)
    return features_paths, pack_sequence(mel_features, enforce_sorted=False)


dataloader = DataLoader(
    dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=8, collate_fn=inference_collate_batch
)
print(f"[Info]: Finish loading data!", flush=True)

num_speakers = 600
probs = np.zeros((len(dataset), num_speakers))
ckpts = ["BiGRU-Drop" + str(i) for i in range(2)]
for i, ckpt_path in enumerate(ckpts):
    print(f"[Info]: {i + 1}-th inference.", flush=True)
    infer_list = []
    model = BiGRUClassifier(embed_size=40, hidden_size=256, num_layers=3, num_classes=num_speakers).to(device)
    model.load_state_dict(torch.load(f"./ckpts/{ckpt_path}.ckpt"))
    model.eval()
    for _, mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            infer_list.append(outs)
    probs += torch.cat(infer_list).cpu().numpy()

feat_paths = []
for feat_path, _ in tqdm(dataloader):
    feat_paths.extend(feat_path)
preds = probs.argmax(1).tolist()
print("len check:", len(feat_paths), len(preds))
results = [["Id", "Category"]]
for feat_path, pred in zip(feat_paths, preds):
    results.append([feat_path, mapping["id2speaker"][str(pred)]])
with open("./submission.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(results)

# model = BiGRUClassifier(embed_size=40, hidden_size=256, num_layers=2, num_classes=num_speakers).to(device)
# model.load_state_dict(torch.load("./ckpts/BiGRU-mean-feat-full_scaling.ckpt"))
# model.eval()
# print(f"[Info]: Finish creating model!", flush=True)
#
# results = [["Id", "Category"]]
# for feat_paths, mels in tqdm(dataloader):
#     with torch.no_grad():
#         mels = mels.to(device)
#         outs = model(mels)
#         preds = outs.argmax(1).cpu().numpy().tolist()
#         for feat_path, pred in zip(feat_paths, preds):
#             results.append([feat_path, mapping["id2speaker"][str(pred)]])
#
# with open("./submission.csv", "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(results)
