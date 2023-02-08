import torch
import pickle
from hummingbird.ml import convert

with open("preprocessor/zscore-full.skl", "rb") as fp:
    scaler = pickle.load(fp)

scaler_hb = convert(scaler, "pytorch")
print(scaler_hb)


class Preprocessor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.data_pipeline = scaler_hb

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.data_pipeline.transform(features)


sample_tensor = torch.randn(500, 40, dtype=torch.float32)
jit_m = torch.jit.trace(Preprocessor(), sample_tensor)
jit_m.save("./scaler.jit")
