from sklearn.base import BaseEstimator, ClassifierMixin, clone
from venn_abers import VennAbersCalibrator

class VennAbersSKlearn(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper around VennAbersCalibrator.

    Parameters
    ----------
    base_estimator : estimator with fit/predict_proba
        e.g., ComplementNB()
    inductive : bool
        True -> IVAP (split variant)
    cal_size : float
        Fraction of training data for the calibration split.
    random_state : int or None
    """
    def __init__(self, estimator=None, inductive=True, cal_size=0.2, random_state=None):
        self.estimator = estimator
        self.inductive = inductive
        self.cal_size = cal_size
        self.random_state = random_state
        self._va = None

    def fit(self, X, y):
        # clone base estimator so each OvR label gets a fresh copy
        est = clone(self.estimator)
        self._va = VennAbersCalibrator(
            estimator=est,
            inductive=self.inductive,
            cal_size=self.cal_size,
            random_state=self.random_state,
        )
        self._va.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._va.predict_proba(X)

    def predict(self, X):
        return self._va.predict(X)
