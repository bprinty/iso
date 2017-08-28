# -*- coding: utf-8 -*-
#
# Learner utilities.
#
# @author <bprinty@gmail.com>
# ----------------------------------------------


# imports
# -------
import os
import re
import numpy
import warnings
from sklearn.base import BaseEstimator, clone
from sklearn.svm import SVC
from sklearn.externals import joblib
from gems import composite

from .transform import TransformChain, Reduce
from .feature import FeatureTransform
from .jade import session


# model building
# --------------
class Learner(BaseEstimator):
    """
    Machine-learning from arbitrary vectorizer. This object also
    allows for swapping different models/features and doing grid
    searches to obtain optimal parameters for models.

    Args:
        transform (Transform, list): Transform to apply to data to convert
            targets and responses into a ai-readable format.
        model (obj): Classifier to use in learning.
    """
    _X = None
    _Y = None

    def __init__(self, transform, model=SVC(), shaper=None, filename=None):
        # use composite transform for everything, so that
        # simulation processors can be skipped during prediction
        if isinstance(transform, TransformChain):
            self.vectorizer = transform
        else:
            self.vectorizer = TransformChain(transform)
        self.shaper = shaper
        self.model = model
        self.filename = filename
        return

    def __repr__(self):
        return '{}(\n\ttransform={},\n\tmodel={},\n\tshaper={})'.format(
            self.__class__.__name__,
            repr(self.vectorizer),
            repr(self.model),
            repr(self.shaper)
        )

    def __copy__(self):
        return self.__class__(
            transform=self.vectorizer.clone(),
            model=clone(self.model),
            shaper=self.shaper.clone()
        )

    def __deepcopy__(self):
        return self.__copy__()

    @classmethod
    def from_config(cls, filename):
        """
        TODO: THIS
        """
        raise NotImplementedError('Loading from config file not currently supported!')

    @classmethod
    def load(cls, filename):
        """
        Load model pickle.

        Args:
            filename (str): Name of known model to load, or filename
                for model pickle or config file.
        """
        global session
        # try loading file directly
        if os.path.exists(filename):
            if filename[-4:] == '.yml' or filename[-5:] == '.yaml':
                return cls.from_config(filename)
            else:
                return joblib.load(filename)
        
        # try loading pickle from models directory
        elif os.path.exists(os.path.join(session.models, filename + '.pkl')):
            return joblib.load(os.path.join(session.models, filename + '.pkl'))

        # try loading config from models directory
        elif os.path.exists(os.path.join(session.models, filename + '.yml')):
            return cls.from_config(os.path.join(session.models, filename + '.yml'))

        else:
            raise AssertionError('Cannot load model. Pickle file does not exist!')
        return

    def save(self, filename=None, archive=False):
        """
        Save learner model to file.

        Args:
            filename (str): Name of known model to save to, or filename
                for model pickle or config file.
            archive (bool): If true, save model in internal models directory,
                using filename as the name of the model. This should be used
                only during model development.
        """
        if self.filename is not None and filename is None:
            filename = self.filename
        if filename is None:
            raise AssertionError('filename argument must be specified to save model.')
        if archive:
            filename = os.path.basename(filename)
            filename = os.path.join(session.models, filename + '.pkl')
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        joblib.dump(
            self.__class__(
                transform=self.vectorizer.clone(),
                model=self.model,
                shaper=self.shaper.clone()
            ), filename
        )
        return

    @property
    def feature_names(self):
        """
        Return feature names if feature transform is part of vectorization.
        """
        if not isinstance(self.vectorizer.transforms[-1], FeatureTransform):
            raise AssertionError('Last transformation must be feature transform to get feature names from learner.')
        return self.vectorizer.transforms[-1].feature_names

    def flatten(self, X, Y=None):
        """
        "Flatten" input, changing dimensionality into
        something conducive to AI model development. In a nutshell,
        this decreases the dimensionality of predictors and responses
        until the response vector is one-dimensional.
        """
        if self.shaper is None:
            self.shaper = TransformChain()
            if Y is not None:
                y = Y[0]
                while isinstance(y, (list, tuple, numpy.ndarray)):
                    y = y[0]
                    self.shaper.add(Reduce())
        return self.shaper.fit_transform(X, Y)

    def inverse_flatten(self, X, Y=None):
        """
        "Inverse flatten" input, changing dimensionality back into
        space that can be back-transformed into something
        human-interpretable.
        """
        return self.shaper.inverse_fit_transform(X, Y)

    def transform(self, X, Y=None):
        """
        Transform input data into ai-ready tensor.
        """
        obj = self.vectorizer.clone()
        X, Y = obj.fit_transform(X, Y, pred=True)
        return X, Y

    def score(self, X, Y=None):
        """
        Apply model scoring function on transformed data.
        """
        tX, tY = self.transform(X, Y)
        return self.model.score(tX, y=tY)

    def fit(self, X, Y):
        """
        Train learner for speicific data indices.
        """
        self.shaper = None
        tX, tY = self.vectorizer.fit_transform(X, Y)
        self._X, self._Y = self.flatten(tX, tY)
        self.model.fit(self._X, self._Y)
        return self

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self._X, self._Y

    def fit_predict(self, X, Y):
        """
        Fit models to data and return prediction.
        """
        self.fit(X, Y)
        if self.vectorizer.has_simulator:
            # we don't want to make predictions on the simulated
            # data, because it's only used to boost the training set
            return self.predict(X)
        else:
            # if we can fully back-transform the data, there's
            # no need to re-do the transformation process
            pY = self.model.predict(self._X)
            fX, fY = self.inverse_flatten(self._X, pY)
            rX, rY = self.vectorizer.inverse_fit_transform(fX, fY)
        return rY

    def predict(self, X):
        """
        Predict results from new data.
        """
        obj = self.vectorizer.clone()
        tX, tY = obj.fit_transform(X, None, pred=True)
        fX, fY = self.flatten(tX, self.vectorizer[-1]._Y if self.shaper is None else None)
        pY = self.model.predict(fX)
        fX, fY = self.inverse_flatten(fX, pY)
        rX, rY = obj.inverse_fit_transform(fX, fY)
        return rY
