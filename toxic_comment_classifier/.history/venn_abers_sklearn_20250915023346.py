# venn_abers_sklearn.py
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from scipy import sparse
import numpy as np
from venn_abers import VennAbersCalibrator

class VennAbersSKlearn(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper around VennAbersCalibrator for a *binary* task.
    Ensures y is 1-D dense and returns 2-column probabilities for compatibility.
    """
    def __init__(self, estimator=None, inductive=True, cal_size=0.2, random_state=None):
        self.estimator = estimator
        self.inductive = inductive
        self.cal_size = cal_size
        self.random_state = random_state
        self._va = None
        self.classes_ = np.array([0, 1])  # enforce binary class order

    def _ensure_1d_dense_labels(self, y):
        if sparse.issparse(y):
            y = y.toarray()
        y = np.asarray(y)
        if y.ndim == 2:
            if y.shape[1] == 1:
                y = y.ravel()
            elif y.shape[0] == 1:
                y = y.reshape(-1)
            else:
                raise ValueError(f"Expected 1-D target; got shape {y.shape}.")
        # cast to int 0/1
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
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        # The package returns p* for the positive class (shape: (n_samples,))
        p_pos = np.asarray(self._va.predict_proba(X)).ravel()
        # Return a 2-column array: [P(class 0), P(class 1)]
        return np.column_stack([1 - p_pos, p_pos])

    def predict(self, X):
        # threshold at 0.5 on the positive-class probas
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
