"""Deep learning training loop — multi-task TabNet with early stopping."""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import MODELS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dl_data(
    x_numeric: np.ndarray,
    x_categorical: list[np.ndarray],
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    """Convert numpy arrays to a PyTorch DataLoader."""
    tensors = [
        torch.tensor(x_numeric, dtype=torch.float32),
        *[torch.tensor(xc, dtype=torch.long) for xc in x_categorical],
        torch.tensor(y_cls, dtype=torch.long),
        torch.tensor(y_reg, dtype=torch.float32),
    ]
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_multitask(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_categorical: int,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
) -> dict[str, list[float]]:
    """Training loop with early stopping and cosine annealing LR."""
    torch.manual_seed(RANDOM_SEED)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "val_cls_acc": [], "val_reg_r2": [],
    }
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_losses = []
        for batch in train_loader:
            x_num = batch[0].to(DEVICE)
            x_cats = [batch[i + 1].to(DEVICE) for i in range(n_categorical)]
            y_cls = batch[n_categorical + 1].to(DEVICE)
            y_reg = batch[n_categorical + 2].to(DEVICE)

            optimizer.zero_grad()
            cls_logits, reg_pred = model(x_num, x_cats)
            total_loss, _, _ = loss_fn(cls_logits, reg_pred, y_cls, y_reg)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(total_loss.item())

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_losses, all_preds, all_true_cls, all_pred_reg, all_true_reg = [], [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x_num = batch[0].to(DEVICE)
                x_cats = [batch[i + 1].to(DEVICE) for i in range(n_categorical)]
                y_cls = batch[n_categorical + 1].to(DEVICE)
                y_reg = batch[n_categorical + 2].to(DEVICE)

                cls_logits, reg_pred = model(x_num, x_cats)
                total_loss, _, _ = loss_fn(cls_logits, reg_pred, y_cls, y_reg)
                val_losses.append(total_loss.item())

                all_preds.extend(cls_logits.argmax(dim=1).cpu().numpy())
                all_true_cls.extend(y_cls.cpu().numpy())
                all_pred_reg.extend(reg_pred.cpu().numpy())
                all_true_reg.extend(y_reg.cpu().numpy())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        cls_acc = np.mean(np.array(all_preds) == np.array(all_true_cls))
        ss_res = np.sum((np.array(all_true_reg) - np.array(all_pred_reg)) ** 2)
        ss_tot = np.sum((np.array(all_true_reg) - np.mean(all_true_reg)) ** 2)
        reg_r2 = 1 - ss_res / max(ss_tot, 1e-8)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_cls_acc"].append(float(cls_acc))
        history["val_reg_r2"].append(float(reg_r2))

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                "Epoch %d/%d — train=%.4f, val=%.4f, cls_acc=%.3f, reg_r2=%.4f, lr=%.2e",
                epoch + 1, epochs, avg_train, avg_val, cls_acc, reg_r2,
                optimizer.param_groups[0]["lr"],
            )

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / "dl_multitask_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
                break

    # Load best weights
    model.load_state_dict(torch.load(MODELS_DIR / "dl_multitask_best.pt", weights_only=True))
    logger.info("Training complete — best val_loss=%.4f", best_val_loss)
    return history
