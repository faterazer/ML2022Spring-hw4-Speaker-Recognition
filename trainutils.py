import sys
from copy import deepcopy
from typing import List, Tuple, Union

import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def test(test_loader: DataLoader, model: nn.Module, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    count = 0
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        count += len(y)
        with torch.no_grad():
            y_hat = model(X)
            loss = criterion(y_hat, y)
            preds = y_hat.argmax(1)
            acc = torch.sum((preds == y).float())

            running_loss += loss.item() * len(y)
            running_acc += acc.item()
    return running_loss / count, running_acc / count


def train(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    valid_steps: int,
    trial_name: str,
) -> None:
    writer = SummaryWriter(f"./logs/{trial_name}")
    train_iterator = iter(train_loader)
    best_accuracy = -1.0

    model.train()
    for step in tqdm(range(total_steps), leave=False, file=sys.stdout):
        try:
            X, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            X, y = next(train_iterator)
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        preds = y_hat.argmax(1)
        batch_loss = loss.item()
        batch_acc = torch.mean((preds == y).float()).item()
        writer.add_scalars("loss", {"train_loss": batch_loss}, global_step=step + 1)
        writer.add_scalars("acc", {"train_acc": batch_acc}, global_step=step + 1)

        if (step + 1) % valid_steps == 0:
            valid_loss, valid_acc = test(valid_loader, model, criterion)
            model.train()
            writer.add_scalars("loss", {"valid_loss": valid_loss}, global_step=step + 1)
            writer.add_scalars("acc", {"valid_acc": valid_acc}, global_step=step + 1)
            tqdm.write(
                f"\n{step + 1} steps - Train loss: {batch_loss} | Train acc: {batch_acc} | Valid loss: {valid_loss} | "
                f"Valid acc: {valid_acc}"
            )

            if valid_acc > best_accuracy:
                best_accuracy = valid_acc
                best_state_dict = model.state_dict()
                torch.save(best_state_dict, f"./ckpts/{trial_name}.ckpt")
                tqdm.write("{} steps: Saving model with acc {:.4f}".format(step + 1, valid_acc))


def kfold_train(
    train_loaders: List[DataLoader],
    valid_loaders: List[DataLoader],
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    valid_steps: int,
    trial_name_or_names: Union[str, List[str]],
) -> None:
    assert len(train_loaders) == len(valid_loaders)
    k = len(train_loaders)
    if isinstance(trial_name_or_names, str):
        trial_names = [trial_name_or_names + str(i) for i in range(k)]
    else:
        assert isinstance(trial_name_or_names, list) and len(trial_name_or_names) == k
        trial_names = trial_name_or_names

    model_state_dict = deepcopy(model.state_dict())
    optimizer_state_dict = deepcopy(optimizer.state_dict())
    for i in range(k):
        tqdm.write("==================== {}-th fold ====================".format(i + 1))
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        train(
            train_loaders[i],
            valid_loaders[i],
            model,
            criterion,
            optimizer,
            total_steps,
            valid_steps,
            trial_names[i],
        )
