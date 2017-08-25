# -*- coding: utf-8 -*-
#
# Additional scikit-learn models that can be useful
# in the context of jade transforms.
#
# @author <bprinty@gmail.com>
# ----------------------------------------------


# imports
# -------
import numpy
from scipy import optimize
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin


# basic models
# ------------
class Threshold(BaseEstimator):
    """
    Simple threshold model for single column. If no threshold is specified,
    this model will use the bounds and step parameters to brute-force
    optimize a thresold that maximizes f1 score for predicting variants.

    Args:
        idx (str): Column to parameterize event-aware estimator for.
        thresh (int): Feature value to use as threshold in calling event.
        bounds (list): Bounds for parameterizing threshold.
        step (float): Step size to use in parameterizing new threshold.
    """
    
    def __init__(self, idx, thresh=0.5, bounds=[0, 1], step=0.001):
        self.idx = idx
        self.thresh = thresh
        self._thresh = thresh
        self.bounds = bounds
        self.step = step
        return

    def reset(self):
        self.thresh = self._thresh
        return

    def fit(self, X, y):
        arr = numpy.array([X[idx][self.idx] for idx in range(0, len(X))])
        opt = optimize.brute(
            lambda x: - metrics.f1_score(y, arr >= x),
            ranges=(slice(self.bounds[0], self.bounds[1], self.step),)
        )
        self.thresh = opt[0]
        return self

    def predict(self, X):
        arr = numpy.array([X[idx][self.idx] for idx in range(0, len(X))])
        return arr >= self.thresh


class MultiThreshold(BaseEstimator):
    """
    TODO: this. Like above but with multiple columns.
    """
    
    def __init__(self, **kwargs):
        self.thresh = kwargs
        self.idx = dict(zip(kwargs.keys(), [None] * len(kwargs)))
        return

    def fit(self, X, y):
        # TODO: enable column-specific threshold parameterization,
        #       that utilizes the thresholds from all columns to optimize
        #       all 'None' threshold values. Maybe use basinhopping?
        # for key in self.thresh:
        #     if self.thresh[key] is None:
        #         arr = numpy.array([X[idx][self.idx] for idx in range(0, len(X))])
        #         opt = optimize.brute(
        #             lambda x: - metrics.f1_score(y, arr >= x),
        #             ranges=(slice(self.bounds[0], self.bounds[1], self.step),)
        #         )
        #         self.thresh = opt[0]
        return self

    def predict(self, X):
        res = numpy.array([True] * len(X), dtype=bool)
        for col in self.thresh.keys():
            cidx = self.idx[col]
            arr = numpy.array([X[idx][cidx] for idx in range(0, len(X))]) >= self.thresh[col]
            res &= arr
        return res


class IdentityModel(BaseEstimator):
    """
    TODO: this. Predict everything.
    """
    def fit(self, X, y):
        return self

    def predict(self, X):
        return [True] * len(X)

    def score(self, X, y, metric=metrics.accuracy_score):
        return metric(self.predict(X), y)
