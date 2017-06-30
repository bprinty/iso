# -*- coding: utf-8 -*-
#
# Mahine learning methods for NDS event detection.
#
# @author <bprinty@gmail.com>
# ----------------------------------------------


# imports
# -------
import os
import re
import numpy
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, ShuffleSplit
from sklearn.exceptions import UndefinedMetricWarning
from gems import composite

from .jade import session


# config
# ------
warnings.simplefilter("ignore", UndefinedMetricWarning)


# helpers
# -------
def flatten(array):
    if len(array) == 0 or isinstance(array, pandas.DataFrame):
        return array
    if isinstance(array[0], (list, tuple, numpy.ndarray, pandas.DataFrame)):
        res = []
        for item in list(array):
            if isinstance(item, pandas.DataFrame):
                res.append(item)
            else:
                res.extend(item)
        array = res
        if isinstance(array[0], pandas.DataFrame):
            array = pandas.concat(array)
            array = array.reset_index(drop=True)
    return array



class Transform(BaseEstimator, TransformerMixin):
    """
    Data transform operator, transforming data from one space
    into another space for downstream processing.
    """
    cache = False
    _X = None
    _y = None

    def hash(X, y):
        X = numpy.array(X)
        y = numpy.array(y) if y is not None else y
        args = numpy.array([(k, v) for k, v in self.__dict__.iteritems() if k[0] != '_'])
        ha = hash(args)
        hx = hash(X.tobytes())
        hy = hash(y.tobytes()) if y is not None else hash(y)
        return '.'.join(ha, hx, hy)

    def save(self, X, y):
        """
        Cache model based on specified inputs. In the future, if the
        model is called with the same inputs and arguments, the transform doesn't
        have to be re-applied.
        """
        global session
        cache = self.hash(X, y)
        dirname = os.path.join(session.data, self.__class__.__name__)
        path = os.path.join(dirname, cache)
        logging.info('Saving {} transformed data to {}.'.format(self.__class__.__name__, path))
        joblib.dump(path, {
            'X': self._X,
            'y': self._y
        })
        return

    def load(self, X, y):
        global session
        cache = self.hash(X, y)
        dirname = os.path.join(session.data, self.__class__.__name__)
        path = os.path.join(dirname, cache)
        logging.info('Loading {} transformed data from {}.'.format(self.__class__.__name__, path))
        if os.path.exists(path):
            data = joblib.load(path)
            return data['X'], data['y']
        else:
            return None

    @property
    def targets(self):
        return self._X

    @property
    def response(self):
        return self._y
    
    def fit(self, X, y=None):
        # configure data
        X = numpy.array(X)
        y = numpy.array(y) if y is not None else y
        
        # load from cache (if specified)
        if self.cache:
            data = self.load(X, y)
            if data is not None:
                self._X, self._y = data
                return

        # TODO: FIGURE OUT HOW TO PARALLELIZE FITTING
        # if jobs == 1:
        #     results = [_fit_vectorizer(*arg) for arg in args]
        # else:
        #     results = joblib.Parallel(n_jobs=jobs)(joblib.delayed(_fit_vectorizer)(*arg) for arg in args)

        # transform
        self._X = []
        self._y = [] if y is not None else y
        for ix, x in enumerate(X):
            if y is None:
                tx, ty = self.transform(x)
            else:
                tx, ty = self.transform(x, y[ix])
                self._y.append(ty)
            self._X.append(tx)
        self._X = numpy.array(self._X)
        self._y = numpy.array(self._y) is y is not None else y
        
        # save to cache (if specified)
        if self.cache:
            self.save(X, y)
        return self

    def inverse_fit(self, X, y=None):
        """
        Inverse-transform targets back to original space
        from transformed space. Since we're backing out of the rabbit
        hole here, we don't need to worry about caching.
        """
        iX = []
        iy = [] if y is not None else y
        for ix, x in enumerate(X):
            if y is None:
                tx, ty = self.inverse_transform(x)
            else:
                tx, ty = self.inverse_transform(x, y[ix])
                iy.append(ty)
            iX.append(tx)
        iX = numpy.array(iX)
        iy = numpy.array(iy) is y is not None else y
        return iX, iy

    def transform(self, X, y=None):
        return X, y

    def inverse_transform(self, X, y=None):
        return X, y


class ComplexTransform(BaseEstimator, TransformerMixin):
    """
    Complex data transform operator requiring two passes through the data - 
    once for a "peek" at the landscape of the data, and another for
    transforming the data into another space.
    """
    registered = False
    
    def register(self, X, y=None):
        return

    def parameterize(self):
        return

    def fit(self, X, y=None):
        # configure data
        X = numpy.array(X)
        y = numpy.array(y) if y is not None else y
        
        # load from cache (if specified)
        if self.cache:
            data = self.load(X, y)
            if data is not None:
                self._X, self._y = data
                return
        
        # register
        for ix, x in enumerate(X):
            if self.registered:
                break
            if y is None:
                self.register(x)
            else:
                self.register(x, y[ix])

        # parameterize
        self.parameterize()

        # transform
        self._X = []
        self._y = [] if y is not None else y
        for ix, x in enumerate(X):
            if y is None:
                tx, ty = self.transform(x)
            else:
                tx, ty = self.transform(x, y[ix])
                self._y.append(ty)
            self._X.append(tx)
        self._X = numpy.array(self._X)
        self._y = numpy.array(self._y) is y is not None else y
        
        # save to cache (if specified)
        if self.cache:
            self.save(X, y)
        return self


# vectorizers
# -----------
class CompositeTransform(Transform):
    """
    Manager for all transpose layers.
    """

    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, Transform):
                raise AssertionError('Inputs to vectorizer must be transform operator!')
        self.transforms = args
        return

    def fit(self, X, y=None):
        # load from cache (if specified)
        if self.cache:
            data = self.load(X, y)
            if data is not None:
                self._X, self._y = data
                return

        # transform
        tx, ty = X, y
        for xf in self.transforms:
            tx, ty = xf.transform(tx, ty)
        self._X = tx, self._y = ty

        # save to cache (if specified)
        if self.cache:
            self.save(X, y)
        return self

    def transform(self, X, y=None):
        self.fit(X, y)
        return self._X, self._y

    def inverse_transform(self, X, y=None):
        tx, ty = X, y
        for xf in self.transforms:
            tx, ty = xf.transform(tx, ty)
        return tx, ty



# feature extraction
# ------------------
class Learner(object):
    """
    Machine-learning from arbitrary vectorizer. This object also
    allows for swapping different models/features and doing grid
    searches to obtain optimal parameters for models.

    Args:
        vectorizer (Vectorizer, list): Vectorizer to use in extracting features, or
            list of features to instantiate vectorizer with.
        model (Classifier): Classifier from sklearn to use in learning.
    """

    def __init__(self, transform, model=MLPClassifier()):
        if isinstance(transform, (list, tuple)):
            transform = CompositeTransform(*transform)
        self.transform = transform
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
                transform=self.transform.clone(),
                model=self.model
            ), filename
        )
        return

    def fit(self, X, y):
        """
        Train learner for speicific data indices.
        """
        tx, ty = self.transform.fit(X, y)
        self.model.fit(tx, ty)
        return self

    def transform(self, X, y=None):
        """
        Transform input data into ai-ready tensor.
        """
        X, y = self.transform.transform(X, y)
        if y is None:
            return X
        return X, y

    def fit_predict(self, X, y):
        """
        Fit models to data and return prediction.
        """
        self.fit(X, y)
        return self.predict(X)

    def predict(self, X):
        """
        Predict results from new data.
        """
        tx = self.transform.transform(X)
        self.model.predict()
        return
