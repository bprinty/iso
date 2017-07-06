# -*- coding: utf-8 -*-
#
# Mahine learning methods for NDS event detection.
#
# @author <bprinty@gmail.com>
# ----------------------------------------------


# imports
# -------
import os
import numpy
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy
import hashlib
import logging

from .jade import session
from .transform import Transform, ComplexTransform


# classes
# -------
class Feature(Transform):
    """
    Data transform operator for feature extraction, allowing
    learner objects to return data frames with extracted features
    and names.
    """
    __depends__ = ()

    @property
    def name(self):
        """
        Return consolidated name of processor. This allows from name
        determination from the __name__, _name and __class__.__name__
        properties, with that heiarchy.
        """
        if hasattr(self, '__name__'):
            return self.__name__
        elif hasattr(self, '_name'):
            return self._name
        else:
            return self.__class__.__name__


class ComplexFeature(ComplexTransform, Feature):
    """
    Data transform operator for two-pass feature extraction, allowing
    learner objects to return data frames with extracted features
    and names.
    """
    pass


# vectorizers
# -----------
class FeatureTransform(ComplexTransform):
    """
    Manager for doing feature extraction for a set of Feature or
    ComplexFeature transformations.
    """

    def __init__(self, *args):
        if len(args) == 0:
            raise AssertionError('No features specified!')
        if isinstance(args[0], (list, tuple)):
            args = args[0]
        self.features = []
        # self.dependencies = []
        for arg in args:
            self.add(arg)
        return

    def add(self, feature):
        """
        Add transform to internal transform list.
        """
        if not isinstance(feature, Feature):
            raise AssertionError('No rule for feature extraction with {} object!'.format(str(type(feature))))
        self.features.append(feature)
        # for dep in feature.__depends__:
        #     if dep.name not in self.dependency_names + self.feature_names:
        #         self.dependencies.append(dep)
        return

    @property
    def feature_names(self):
        return map(lambda x: x.name, self.features)

    # @property
    # def dependency_names(self):
    #     return map(lambda x: x.name, self.dependencies)

    @property
    def complex(self):
        for feat in self.features:
            if isinstance(feat, ComplexFeature):
                return True
        return False

    @property
    def registered(self):
        for feat in self.features:
            if not feat.registered:
                return False
        return True

    def register(self, x, y=None):
        """
        Register operator for features. The method is only called with
        one input here, because feature transformations are truth-agnostic,
        producing truth in the same dimensionality.
        """
        for feat in self.features:
            if isinstance(feat, ComplexFeature):
                # only called with x because feature transformations
                # are truth-agnostic, producing truth with the same
                # dimensionality
                feat.register(x)
        return

    def parameterize(self):
        """
        Parameterize operator for features. The method is only called with
        one input here, because feature transformations are truth-agnostic,
        producing truth in the same dimensionality.
        """
        for feat in self.features:
            if isinstance(feat, ComplexFeature):
                feat.parameterize()
        return

    def transform(self, x, y=None):
        """
        Transform operator for features.
        """
        X = []
        for feat in self.features:
            # only called with x because feature transformations
            # are truth-agnostic, producing truth with the same
            # dimensionality
            tx = feat.transform(x)
            X.append(tx)
        return numpy.array(X), y
