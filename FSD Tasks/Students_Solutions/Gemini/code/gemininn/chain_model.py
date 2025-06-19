import random
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Tuple
from scipy.stats import skew, kurtosis
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline



class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns].values

class SensorSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.stack(X['sensor_data'].values)

class SensorScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = RobustScaler()

    def fit(self, X, y=None):
        n_samples, n_steps, n_sensors = X.shape
        flat = X.reshape(-1, n_sensors)
        self.scaler.fit(flat)
        return self

    def transform(self, X):
        n_samples, n_steps, n_sensors = X.shape
        flat = X.reshape(-1, n_sensors)
        flat_scaled = self.scaler.transform(flat)
        return flat_scaled.reshape(n_samples, n_steps, n_sensors)

class SensorFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fft_features = np.fft.rfft(X, axis=1).real
        fft_mean = fft_features.mean(axis=1)
        fft_std = fft_features.std(axis=1)
        features = np.concatenate([
            X.mean(axis=1),
            X.std(axis=1),
            X.max(axis=1),
            X.min(axis=1),
            skew(X, axis=1),
            kurtosis(X, axis=1),
            fft_mean,
            fft_std
        ], axis=1)
        return features

class MultilabelOversampler(BaseEstimator):
    def __init__(self, target_per_class: int = 1000):
        self.target_per_class = target_per_class

    def fit_resample(self, X, y):
        # X: array-like, y: binary indicator matrix
        y_tuples = [tuple(row) for row in y]
        unique, counts = np.unique(y_tuples, axis=0, return_counts=True)
        X_aug, y_aug = [], []
        for label, count in zip(unique, counts):
            mask = np.all(y == label, axis=1)
            idx = np.where(mask)[0]
            n_needed = self.target_per_class - count
            if n_needed <= 0:
                continue
            for _ in range(n_needed):
                i = np.random.choice(idx)
                X_aug.append(X[i])
                y_aug.append(y[i])
        if not X_aug:
            return X, y
        return np.vstack([X, np.array(X_aug)]), np.vstack([y, np.array(y_aug)])

class ChainEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator=None,
        n_chains: int = 5,
        thresholds: Tuple[float, ...] = (0.3, 0.3, 0.25, 0.2),
        random_state: int = 42
    ):
        self.base_estimator = base_estimator
        self.n_chains = n_chains
        self.thresholds = np.array(thresholds)
        self.random_state = random_state

    def fit(self, X, y):
        # Create and fit multiple classifier chains
        self.chains_ = []
        for i in range(self.n_chains):
            est = deepcopy(self.base_estimator)
            chain = ClassifierChain(est, order='random', random_state=self.random_state + i)
            chain.fit(X, y)
            self.chains_.append(chain)
        return self

    def predict_proba(self, X):
        # Average predicted probabilities across chains
        avg_proba = sum(chain.predict_proba(X) for chain in self.chains_) / len(self.chains_)
        return avg_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > self.thresholds).astype(int)

# Function to convert integer labels to multilabel tuples

def label_to_tuple(label: int) -> Tuple[int, int, int, int]:
    mapping = {
        0: (0, 0, 0, 0),
        1: (0, 0, 1, 0),
        2: (0, 0, 0, 1),
        3: (0, 0, 1, 1),
        4: (1, 0, 0, 0),
        5: (0, 1, 0, 0),
        6: (1, 1, 0, 0),
        7: (1, 0, 1, 0),
        8: (0, 1, 0, 1),
        9: (1, 0, 0, 1),
        10: (0, 1, 1, 0),
        11: (1, 1, 1, 1)
    }
    return mapping[label]


def build_pipeline(tab_columns, target_per_class: int = 1000):
    tab_pipeline = Pipeline([
        ('selector', DataFrameSelector(tab_columns)),
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    sensor_pipeline = Pipeline([
        ('selector', SensorSelector()),
        ('scaler', SensorScaler()),
        ('features', SensorFeatureTransformer())
    ])

    preprocessor = FeatureUnion([
        ('tab', tab_pipeline),
        ('sensor', sensor_pipeline)
    ])

    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('oversampler', MultilabelOversampler(target_per_class=target_per_class)),
        ('classifier', ChainEnsemble(
            base_estimator=XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=5
            ),
            n_chains=5,
            thresholds=(0.3, 0.3, 0.25, 0.2),
            random_state=42
        ))
    ])
    return pipeline