import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence, unpack_sequence


class BiGRUClassifier(nn.Module):
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
        num_classes: int = 600,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.5,
            bidirectional=True,
            batch_first=True,
        )
        self.gru.flatten_parameters()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size * 4, num_classes),
        )

    def forward(self, X: PackedSequence) -> Tensor:
        outputs, _ = self.gru(X)
        # outputs = torch.stack([torch.mean(o, dim=0) for o in unpack_sequence(outputs)])
        # return self.fc(outputs)
        outputs = [self.fc(o) for o in unpack_sequence(outputs)]
        outputs = [torch.mean(o, dim=0) for o in outputs]
        return torch.stack(outputs)
