# -*- coding: utf-8 -*-
#
# Additional scikit-learn models that can be useful
# in the context of iso transforms.
#
# @author <bprinty@gmail.com>
# ----------------------------------------------


# imports
# -------
import numpy
# from scipy import optimize
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin


# basic models
# ------------
class Threshold(BaseEstimator):
    """
    TODO: this. threshold using single feature column -- this requires
          that a FeatureTransform has been applied as the last step of
          the transform.
    """
    pass


class MultiThreshold(BaseEstimator):
    """
    TODO: this. Like above but with multiple columns.
    """
    pass


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
