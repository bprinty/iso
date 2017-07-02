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
from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy

from .jade import session


# classes
# -------
class Transform(BaseEstimator, TransformerMixin):
    """
    Data transform operator, transforming data from one space
    into another space for downstream processing.
    """
    cache = False
    jobs = 1
    _X, _Y = None, None
    _iX, _iY = None, None

    def clone(self):
        """
        Produce "copy" of transform that can be used fit data
        outside of the current context.
        """
        ret = deepcopy(self)
        ret._X, ret._Y = None, None
        ret._iX, ret._iY = None, None
        return ret

    def hash(self, X, Y=None):
        """
        Generate unique hash based on data and truth.

        Args:
            X (numpy.array): Array with targets to apply transformation to.
            Y (numpy.array): Array with responses to apply transformation to.
        """
        X = numpy.array(X)
        Y = numpy.array(Y) if Y is not None else Y
        args = numpy.array([(k, v) for k, v in self.__dict__.iteritems() if k[0] != '_'])
        ba = args.tobytes()
        bx = X.tobytes()
        by = Y.tobytes() if Y is not None else ''
        return abs(hash(ba + bx + by))

    def save(self, X, Y=None):
        """
        Cache model based on specified inputs. In the future, if the
        model is called with the same inputs and arguments, the transform doesn't
        have to be re-applied.

        Args:
            X (numpy.array): Array with targets to apply transformation to.
            Y (numpy.array): Array with responses to apply transformation to.
        """
        global session
        cache = str(self.hash(X, Y))
        dirname = os.path.join(session.data, self.__class__.__name__)
        path = os.path.join(dirname, cache)
        logging.info('Saving {} transformed data to {}.'.format(self.__class__.__name__, path))
        joblib.dump(path, {
            'X': self._X,
            'Y': self._Y
        })
        return

    def load(self, X, Y=None):
        """
        Load model based on specified inputs. If the model has been previously used
        to transform data with the same inputs and configuration, then we can
        shortcut the training process.

        Args:
            X (numpy.array): Array with targets to apply transformation to.
            Y (numpy.array): Array with responses to apply transformation to.
        """
        global session
        cache = str(self.hash(X, Y))
        dirname = os.path.join(session.data, self.__class__.__name__)
        path = os.path.join(dirname, cache)
        logging.info('Loading {} transformed data from {}.'.format(self.__class__.__name__, path))
        if os.path.exists(path):
            data = joblib.load(path)
            return data['X'], data['Y']
        else:
            return None

    @property
    def targets(self):
        """
        Return the targets that the transform produced. This property
        will be set with new data onece the fit() method has been called.
        """
        return self._X

    @property
    def inverse_targets(self):
        """
        Return the targets that the transform produced. This property
        will be set with new data onece the fit() method has been called.
        """
        return self._iX

    @property
    def responses(self):
        """
        Return the reponse that the transform produced. This property
        will be set with new data onece the fit() method has been called.
        """
        return self._Y

    @property
    def inverse_responses(self):
        """
        Return the reponse that the transform produced. This property
        will be set with new data onece the fit() method has been called.
        """
        return self._iY
    
    def fit(self, X, Y=None):
        """
        Wrapper around transformations, doing the transformation process
        for each input in the target and response vectors.

        Args:
            X (numpy.array): Array with targets to apply transformation to.
            Y (numpy.array): Array with responses to apply transformation to.
        """
        # configure data
        X = numpy.array(X)
        Y = numpy.array(Y) if Y is not None else Y
        
        # load from cache (if specified)
        if self.cache:
            data = self.load(X, Y)
            if data is not None:
                self._X, self._Y = data
                return

        # TODO: FIGURE OUT HOW TO PARALLELIZE FITTING
        # if jobs == 1:
        #     results = [_fit_vectorizer(*arg) for arg in args]
        # else:
        #     results = joblib.Parallel(n_jobs=jobs)(joblib.delayed(_fit_vectorizer)(*arg) for arg in args)

        # transform
        self._X = []
        self._Y = [] if Y is not None else Y
        for ix, x in enumerate(X):
            if Y is None:
                tx, ty = self.transform(x)
            else:
                # the index for Y needs to be capped, because when
                # vectorization is applied in the context of prediction
                # via learner, we can't guarantee that the predictors
                # and original truth are the same length
                tx, ty = self.transform(x, Y[min(ix, len(Y) - 1)])
                self._Y.append(ty)
            self._X.append(tx)
        self._X = numpy.array(self._X)
        self._Y = numpy.array(self._Y) if Y is not None else Y
        
        # assert that predictor and response have
        # same outer dimensionality
        if Y is not None and len(self._X) != len(self._Y):
            raise AssertionError('Error: Target and response vectors must have the same outer dimensionality!')
        
        # save to cache (if specified)
        if self.cache:
            self.save(X, Y)
        return self

    def fit_transform(self, X, Y=None):
        """
        Fit and transform full input target space.

        Args:
            X (numpy.array): Array with targets to apply transformation to.
            Y (numpy.array): Array with responses to apply transformation to.
        """
        self.fit(X, Y)
        return self._X, self._Y

    def inverse_fit(self, X, Y=None):
        """
        Inverse-transform targets back to original space
        from transformed space. Since we're backing out of the rabbit
        hole here, we don't need to worry about caching.

        Args:
            X (numpy.array): Array with targets to apply transformation to.
            Y (numpy.array): Array with responses to apply transformation to.
        """
        if self._X is None:
            raise AssertionError('Transform must have initial fit to inverse-transform!')
        self._iX = []
        self._iY = [] if Y is not None else Y
        for ix, x in enumerate(X):
            if Y is None:
                tx, ty = self.inverse_transform(x)
            else:
                tx, ty = self.inverse_transform(x, Y[min(ix, len(Y) - 1)])
                self._iY.append(ty)
            self._iX.append(tx)
        self._iX = numpy.array(self._iX)
        self._iY = numpy.array(self._iY) if Y is not None else Y
        return self

    def inverse_fit_transform(self, X, Y=None):
        """
        Fit and transform full input target space.

        Args:
            X (numpy.array): Array with targets to apply inverse transformation to.
            Y (numpy.array): Array with responses to apply inverse transformation to.
        """
        self.inverse_fit(X, Y)
        return self._iX, self._iY

    def transform(self, x, y=None):
        """
        Apply transformation to a single element in target space.

        Args:
            x (object): Single target to apply transformation to.
            y (object): Single response to apply transformation to.
        """
        return x, y

    def inverse_transform(self, x, y=None):
        """
        Apply inverse transformation to a single element in target space.

        Args:
            x (object): Single target to apply transformation to.
            y (object): Single response to apply transformation to.
        """
        return x, y


class ComplexTransform(BaseEstimator, TransformerMixin):
    """
    Complex data transform operator requiring two passes through the data - 
    once for a "peek" at the landscape of the data, and another for
    transforming the data into another space.
    """
    registered = False
    
    def register(self, x, y=None):
        """
        Register a single element in target space.

        Args:
            x (object): Single target to apply transformation to.
            y (object): Single response to apply transformation to.
        """
        return

    def parameterize(self):
        """
        Parameterize models for downstream transformation.
        """
        return

    def fit(self, X, Y=None):
        # configure data
        X = numpy.array(X)
        Y = numpy.array(Y) if Y is not None else Y
        
        # load from cache (if specified)
        if self.cache:
            data = self.load(X, Y)
            if data is not None:
                self._X, self._Y = data
                return
        
        # register
        for ix, x in enumerate(X):
            if self.registered:
                break
            if Y is None:
                self.register(x)
            else:
                self.register(x, Y[min(ix, len(Y) - 1)])

        # parameterize
        self.parameterize()

        # transform
        self._X = []
        self._Y = [] if Y is not None else Y
        for ix, x in enumerate(X):
            if Y is None:
                tx, ty = self.transform(x)
            else:
                tx, ty = self.transform(x, Y[min(ix, len(Y) - 1)])
                self._Y.append(ty)
            self._X.append(tx)
        self._X = numpy.array(self._X)
        self._Y = numpy.array(self._Y) if Y is not None else Y
        
        # save to cache (if specified)
        if self.cache:
            self.save(X, Y)
        return self


# vectorizers
# -----------
class CompositeTransform(Transform):
    """
    Manager for all transpose layers.
    """

    def __init__(self, *args):
        if len(args) == 0:
            raise AssertionError('No transforms specified!')
        if isinstance(args[0], (list, tuple)):
            args = args[0]
        self.transforms = args
        return

    def add(self, transform):
        if not isinstance(arg, Transform):
            raise AssertionError('Inputs to vectorizer must be transform operator!')
        self.transforms.append(arg)
        return

    def fit(self, X, Y=None):
        # load from cache (if specified)
        if self.cache:
            data = self.load(X, Y)
            if data is not None:
                self._X, self._Y = data
                return

        # fit individual transforms
        tx, ty = X, Y
        for xf in self.transforms:
            tx, ty = xf.fit_transform(tx, ty)
        self._X, self._Y = tx, ty

        # save to cache (if specified)
        if self.cache:
            self.save(X, Y)
        return self

    def inverse_fit(self, X, Y=None):
        # inverse fit individual transforms
        tx, ty = X, Y
        for xf in self.transforms:
            tx, ty = xf.inverse_fit_transform(tx, ty)
        self._iX, self._iY = tx, ty
        return self
