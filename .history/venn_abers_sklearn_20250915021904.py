# venn_abers_sklearn.py
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from scipy import sparse
import numpy as np
from venn_abers import VennAbersCalibrator

class VennAbersSKlearn(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper around VennAbersCalibrator.

    Parameters
    ----------
    estimator : estimator with fit/predict_proba (e.g., ComplementNB())
    inductive : bool  (True -> IVAP split variant)
    cal_size : float  (fraction of training data used for calibration)
    random_state : int or None
    """
    def __init__(self, estimator=None, inductive=True, cal_size=0.2, random_state=None):
        self.estimator = estimator
        self.inductive = inductive
        self.cal_size = cal_size
        self.random_state = random_state
        self._va = None

    def _to_1d_dense(self, y):
        # Accept pandas Series/DF, numpy arrays, or scipy sparse
        if hasattr(y, "values"):
            y = y.values
        if sparse.issparse(y):
            y = y.toarray()
        y = np.asarray(y)
        # y can be shape (n_samples, 1) from OvR; ravel to 1-D
        return y.ravel()

    def fit(self, X, y):
        y = self._to_1d_dense(y)  # <-- critical fix
        est = clone(self.estimator)
        self._va = VennAbersCalibrator(
            estimator=est,
            inductive=self.inductive,
            cal_size=self.cal_size,
            random_state=self.random_state,
        )
        self._va.fit(X, y)
        # Expose classes_ for sklearn meta-estimators
        self.classes_ = getattr(self._va, "classes_", np.array([0, 1]))
        return self

    def predict_proba(self, X):
        p = self._va.predict_proba(X)
        p = np.asarray(p)
        # Ensure shape (n_samples, 2) for binary, which OvR expects
        if p.ndim == 1:
            p = np.c_[1.0 - p, p]
        return p

    def predict(self, X):
        return self._va.predict(X)
