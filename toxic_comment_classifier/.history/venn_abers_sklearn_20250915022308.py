# venn_abers_sklearn.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from venn_abers import VennAbersCalibrator

class VennAbersSKlearn(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper around VennAbersCalibrator (IVAP/CVAP).

    Parameters
    ----------
    estimator : estimator with fit/predict_proba (e.g., ComplementNB)
    inductive : bool  (True -> IVAP)
    cal_size : float  (fraction used for calibration split)
    random_state : int or None
    """
    def __init__(self, estimator=None, inductive=True, cal_size=0.2, random_state=None):
        self.estimator = estimator
        self.inductive = inductive
        self.cal_size = cal_size
        self.random_state = random_state
        self._va = None
        self.classes_ = None

    def fit(self, X, y):
        est = clone(self.estimator)
        self._va = VennAbersCalibrator(
            estimator=est,
            inductive=self.inductive,
            cal_size=self.cal_size,
            random_state=self.random_state,
        )
        self._va.fit(X, y)
        # Expose sklearn-ish attribute
        # (OneVsRest doesn't strictly require it, but it helps consistency)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        proba = self._va.predict_proba(X)
        proba = np.asarray(proba)
        # Ensure 2D output for binary tasks: (n_samples, 2)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return proba

    def predict(self, X):
        return self._va.predict(X)
