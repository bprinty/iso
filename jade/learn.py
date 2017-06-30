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
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from gems import composite

from .transform import CompositeTransform


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

    def __init__(self, transform, model=MLPClassifier()):
        if isinstance(transform, (list, tuple)):
            transform = CompositeTransform(*transform)
        self.vectorizer = transform
        self.model = model
        return

    @classmethod
    def from_config(cls, filename):
        """
        TODO: THIS
        """
        return

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
            try:
                return joblib.load(filename)
            except:
                return cls.from_config(filename)
        
        # try loading pickle from models directory
        elif os.path.exists(os.path.join(session.models, filename + '.pkl')):
            return joblib.load(os.path.join(session.models, filename + '.pkl'))

        # try loading config from models directory
        elif os.path.exists(os.path.join(session.models, filename + '.yml')):
            return cls.from_config(os.path.join(session.models, filename + '.yml'))

        else:
            raise AssertionError('Cannot load model. Pickle file does not exist!')
        return

    def save(self, filename, archive=False):
        """
        Save learner model to file.

        Args:
            filename (str): Name of known model to save to, or filename
                for model pickle or config file.
            archive (bool): If true, save model in internal models directory,
                using filename as the name of the model. This should be used
                only during model development.
        """
        if archive:
            filename = os.path.basename(filename)
            filename = os.path.join(session.models, filename + '.pkl')
        joblib.dump(
            self.__class__(
                transform=self.vectorizer.clone(),
                model=self.model
            ), filename
        )
        return

    def transform(self, X, Y=None):
        """
        Transform input data into ai-ready tensor.
        """
        obj = self.vectorizer.clone()
        X, Y = obj.fit_transform(X, Y)
        return X, Y

    def fit(self, X, Y):
        """
        Train learner for speicific data indices.
        """
        tX, tY = self.vectorizer.fit_transform(X, Y)

        fX, fY = [], []
        for i in range(0, len(tY)):
            if isinstance(tY[i], (list, tuple, numpy.ndarray)):
                for j in range(0, len(tY[i])):
                    if isinstance(tY[i][j], (list, tuple, numpy.ndarray)):
                        for k in range(0, len(tY[i][j])):
                            fY.append(tY[i][j][k])
                            fX.append(tX[i][j][k])
                    else:
                        fY.append(tY[i][j])
                        fX.append(tX[i][j])
            else:
                fY.append(tY[i])
                fX.append(tX[i])
        fX, fY = numpy.array(fX), numpy.array(fY)
        # print fX, fY
        # print fX, fY
        # flatten ...
        self.model.fit(fX, fY)
        return self

    def fit_predict(self, X, Y):
        """
        Fit models to data and return prediction.
        """
        self.fit(X, Y)
        return self.predict(X)

    def predict(self, X):
        """
        Predict results from new data.
        """
        obj = self.vectorizer.clone()
        tX, tY = obj.fit_transform(X)
        # flatten ...
        tx, ty = obj.inverse_fit_transform(tx, ty)
        return ty
