# venn_abers_sklearn.py
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from scipy import sparse
import numpy as np
from venn_abers import VennAbersCalibrator

class VennAbersSKlearn(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper around VennAbersCalibrator.
    Expects a binary target per fit call (OvR takes care of looping labels).
    """
    def __init__(self, estimator=None, inductive=True, cal_size=0.2, random_state=None):
        self.estimator = estimator
        self.inductive = inductive
        self.cal_size = cal_size
        self.random_state = random_state
        self._va = None

    def _ensure_1d_dense_labels(self, y):
        # Accepts Series/ndarray/sparse column; returns 1-D np.ndarray
        if sparse.issparse(y):
            y = y.toarray()
        y = np.asarray(y)
        if y.ndim == 2:
            if y.shape[1] == 1:
                y = y.ravel()
            elif y.shape[0] == 1:
                y = y.reshape(-1)
            else:
                raise ValueError(
                    f"VennAbersSKlearn expects a 1-D target per binary task, got shape {y.shape}."
                )
        # make sure dtype is numeric (0/1 for multilabel)
        if not np.issubdtype(y.dtype, np.number):
            y = y.astype(int)
        return y

    def fit(self, X, y):
        y = self._ensure_1d_dense_labels(y)
        base = clone(self.estimator)
        self._va = VennAbersCalibrator(
            estimator=base,
            inductive=self.inductive,
            cal_size=self.cal_size,
            random_state=self.random_state,
        )
        self._va.fit(X, y)
        # follow sklearn convention
        self.classes_ = getattr(self._va, "classes_", np.unique(y))
        return self

    def predict_proba(self, X):
        return self._va.predict_proba(X)

    def predict(self, X):
        return self._va.predict(X)
