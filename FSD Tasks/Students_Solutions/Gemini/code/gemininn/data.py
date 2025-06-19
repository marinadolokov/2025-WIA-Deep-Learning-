import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Sampler

class SensorDataset(Dataset):
    def __init__(self, X_sens: np.ndarray, X_aux: np.ndarray, y: np.ndarray):
        self.X_sens = X_sens.astype(np.float32)
        self.X_aux = X_aux.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X_sens[idx], self.X_aux[idx]), self.y[idx]

# --- Preprocessing helpers ---

def stack_array_column(
    df: pd.DataFrame,
    column: str,
    add_channel: bool = True,
    squeeze: bool = False
) -> np.ndarray:
   
    arr = np.stack(df[column].values)
    if add_channel:
        arr = arr[:, None, :, :]
    if squeeze and add_channel:
        arr = arr.squeeze(axis=1)
    return arr.astype(np.float32)


def extract_features(df: pd.DataFrame, columns: list) -> np.ndarray:

    return df[columns].to_numpy(dtype=np.float32)


def build_standard_loaders(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    sensor_key: str,
    aux_cols: list,
    batch_size: int = 32,
    train_sampler: Sampler = None
) -> dict:

    # Preprocess arrays
    X_tr_s = stack_array_column(X_train, sensor_key)
    X_val_s = stack_array_column(X_val, sensor_key)
    X_te_s = stack_array_column(X_test, sensor_key)
    X_tr_aux = extract_features(X_train, aux_cols)
    X_val_aux = extract_features(X_val, aux_cols)
    X_te_aux = extract_features(X_test, aux_cols)

    # Create datasets
    train_ds = SensorDataset(X_tr_s, X_tr_aux, y_train)
    val_ds   = SensorDataset(X_val_s, X_val_aux, y_val)
    test_ds  = SensorDataset(X_te_s, X_te_aux, y_test)

    # Train loader
    if train_sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Validation and test loaders
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def build_extended_loaders(
    base_loaders: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    ext_aux_cols: list,
    batch_size: int = 32,
    train_sampler: Sampler = None
) -> dict:

    # Preprocess extended aux
    X_tr_ext = extract_features(X_train, ext_aux_cols)
    X_val_ext = extract_features(X_val, ext_aux_cols)
    X_te_ext = extract_features(X_test, ext_aux_cols)
    # Reuse sensor arrays
    X_tr_s = base_loaders['train'].dataset.X_sens
    X_val_s = base_loaders['val'].dataset.X_sens
    X_te_s = base_loaders['test'].dataset.X_sens

    # Build datasets
    train_ext_ds = SensorDataset(X_tr_s, X_tr_ext, y_train)
    val_ext_ds   = SensorDataset(X_val_s, X_val_ext, y_val)
    test_ext_ds  = SensorDataset(X_te_s, X_te_ext, y_test)

    # Train_ext loader
    if train_sampler is not None:
        train_ext = DataLoader(train_ext_ds, batch_size=batch_size, sampler=train_sampler)
    else:
        train_ext = DataLoader(train_ext_ds, batch_size=batch_size, shuffle=True)

    val_ext  = DataLoader(val_ext_ds, batch_size=batch_size, shuffle=False)
    test_ext = DataLoader(test_ext_ds, batch_size=batch_size, shuffle=False)

    return {
        'train_ext': train_ext,
        'val_ext': val_ext,
        'test_ext': test_ext
    }


def build_pers_loaders(
    base_loaders: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    pers_key: str,
    aux_cols: list,
    ext_aux_cols: list = None,
    batch_size: int = 32,
    train_sampler: Sampler = None
) -> dict:
    """
    Build loaders for pers branch and optional extended pers.
    Accepts custom train_sampler.
    """
    # Preprocess pers arrays
    X_tr_p = stack_array_column(X_train, pers_key, squeeze=True)
    X_val_p = stack_array_column(X_val, pers_key, squeeze=True)
    X_te_p  = stack_array_column(X_test, pers_key, squeeze=True)
    # Aux features
    X_tr_aux = extract_features(X_train, aux_cols)
    X_val_aux = extract_features(X_val, aux_cols)
    X_te_aux  = extract_features(X_test, aux_cols)

    # Datasets
    train_p_ds = SensorDataset(X_tr_p, X_tr_aux, y_train)
    val_p_ds   = SensorDataset(X_val_p, X_val_aux, y_val)
    test_p_ds  = SensorDataset(X_te_p, X_te_aux, y_test)

    # Train_p loader
    if train_sampler is not None:
        train_p = DataLoader(train_p_ds, batch_size=batch_size, sampler=train_sampler)
    else:
        train_p = DataLoader(train_p_ds, batch_size=batch_size, shuffle=True)

    val_p  = DataLoader(val_p_ds, batch_size=batch_size, shuffle=False)
    test_p = DataLoader(test_p_ds, batch_size=batch_size, shuffle=False)

    results = {
        'train_pers': train_p,
        'val_pers': val_p,
        'test_pers': test_p
    }

    # Extended pers if any
    if ext_aux_cols:
        X_tr_ext = extract_features(X_train, ext_aux_cols)
        X_val_ext = extract_features(X_val, ext_aux_cols)
        X_te_ext  = extract_features(X_test, ext_aux_cols)
        train_ext_p_ds = SensorDataset(X_tr_p, X_tr_ext, y_train)
        val_ext_p_ds   = SensorDataset(X_val_p, X_val_ext, y_val)
        test_ext_p_ds  = SensorDataset(X_te_p, X_te_ext, y_test)
        if train_sampler is not None:
            train_ext_p = DataLoader(train_ext_p_ds, batch_size=batch_size, sampler=train_sampler)
        else:
            train_ext_p = DataLoader(train_ext_p_ds, batch_size=batch_size, shuffle=True)
        results.update({
            'train_ext_pers': train_ext_p,
            'val_ext_pers': DataLoader(val_ext_p_ds, batch_size=batch_size, shuffle=False),
            'test_ext_pers': DataLoader(test_ext_p_ds, batch_size=batch_size, shuffle=False)
        })
    return results


def build_all_loaders(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    sensor_key: str,
    aux_cols: list,
    ext_aux_cols: list = None,
    pers_key: str = None,
    batch_size: int = 32,
    train_sampler: Sampler = None
) -> dict:

    loaders = build_standard_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        sensor_key, aux_cols, batch_size, train_sampler
    )
    # Extended aux
    if ext_aux_cols:
        loaders.update(
            build_extended_loaders(
                loaders, X_train, y_train, X_val, y_val, X_test, y_test,
                ext_aux_cols, batch_size, train_sampler
            )
        )
    # Pers branch
    if pers_key:
        loaders.update(
            build_pers_loaders(
                loaders, X_train, y_train, X_val, y_val, X_test, y_test,
                pers_key, aux_cols, ext_aux_cols, batch_size, train_sampler
            )
        )
    return loaders


class NumpyDataset(Dataset):
    def __init__(self, X, y, transform=None):
 
        assert len(X) == len(y), "Features and labels must have the same length"
        # Ensure channel dimension exists
        if X.ndim == 3:
            X = X[:, None, :, :]
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img = self.X[idx]  # shape (1, H, W)
        if self.transform:
            img = self.transform(img)
        label = self.y[idx]
        return img, label
