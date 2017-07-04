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
        args = numpy.array([(k, v) for k, v in list(self.__dict__.items()) if k[0] != '_'], dtype=str)
        return joblib.hash([args, X, Y], hash_name='md5')

    def save(self, X, Y=None, filename=None):
        """
        Cache model based on specified inputs. In the future, if the
        model is called with the same inputs and arguments, the transform doesn't
        have to be re-applied.

        Args:
            X (numpy.array): Array with targets to apply transformation to.
            Y (numpy.array): Array with responses to apply transformation to.
        """
        if filename is None:
            global session
            cache = str(self.hash(X, Y))
            dirname = os.path.join(session.data, self.__class__.__name__)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            path = os.path.join(dirname, cache)
        else:
            path = filename
        logging.info('Saving {} transformed data to {}.'.format(self.__class__.__name__, path))
        joblib.dump({
            'X': self._X,
            'Y': self._Y
        }, path)
        return

    def load(self, X, Y=None, filename=None):
        """
        Load model based on specified inputs. If the model has been previously used
        to transform data with the same inputs and configuration, then we can
        shortcut the training process.

        Args:
            X (numpy.array): Array with targets to apply transformation to.
            Y (numpy.array): Array with responses to apply transformation to.
        """
        if filename is None:
            global session
            cache = str(self.hash(X, Y))
            dirname = os.path.join(session.data, self.__class__.__name__)
            path = os.path.join(dirname, cache)
        else:
            path = filename
        logging.info('Loading {} transformed data from {}.'.format(self.__class__.__name__, path))
        if os.path.exists(path):
            data = joblib.load(path)
            return data['X'], data['Y']
        else:
            return None

    @property
    def X(self):
        """
        Return the targets that the transform produced. This property
        will be set with new data onece the fit() method has been called.
        """
        return self._X

    @property
    def iX(self):
        """
        Return the targets that the transform produced. This property
        will be set with new data onece the fit() method has been called.
        """
        return self._iX

    @property
    def Y(self):
        """
        Return the reponse that the transform produced. This property
        will be set with new data onece the fit() method has been called.
        """
        return self._Y

    @property
    def iY(self):
        """
        Return the reponse that the transform produced. This property
        will be set with new data onece the fit() method has been called.
        """
        return self._iY

    def add_block(self, dest, block):
        """
        Add singular "block" to internal variable representing
        either data or truth vectors. By abstracting this into a function,
        it enables other types of processors (i.e. Simulators) to more
        easily change fitting functionality.
        """
        self.__dict__['_{}'.format(dest)].append(block)
        return
    
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
                self.add_block('Y', ty)
            self.add_block('X', tx)
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
                self.add_block('iY', ty)
            self.add_block('iX', tx)
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
                self.add_block('Y', ty)
            self.add_block('X', tx)
        self._X = numpy.array(self._X)
        self._Y = numpy.array(self._Y) if Y is not None else Y
        
        # save to cache (if specified)
        if self.cache:
            self.save(X, Y)
        return self


class Simulator(Transform):
    """
    Transform for simulating new data for model building. These
    transformations don't change the dimensionality of data, they
    just allow for data augmentation for fitting operations.
    """

    def add_block(self, dest, block):
        if not isinstance(block, (list, tuple, numpy.ndarray)):
            raise AssertionError('Error: Simulation transform methods must produce vector type!')
        self.__dict__['_{}'.format(dest)].extend(block)
        return

    def inverse_fit(self, X, Y=None):
        """
        For simulation, no inverse transform is necessary, since
        data already have the dimensionality they need.

        Args:
            X (numpy.array): Array with targets to apply transformation to.
            Y (numpy.array): Array with responses to apply transformation to.
        """
        self._iX, self._iY = X, Y
        return self

    def transform(self, x, y=None):
        """
        Apply transformation to a single element in target space.

        Args:
            x (object): Single target to apply transformation to.
            y (object): Single response to apply transformation to.
        """
        return [x], [y]


class ComplexSimulator(ComplexTransform):
    """
    Transform for simulating new data for model building in a way
    that requires a peak at the data before data are simulated. These
    transformations don't change the dimensionality of data, they
    just 
    """
    def add_block(self, dest, block):
        if not isinstance(block, (list, tuple, numpy.ndarray)):
            raise AssertionError('Error: Simulation transform methods must produce vector type!')
        self.__dict__['_{}'.format(dest)].extend(block)
        return

    def inverse_fit(self, X, Y=None):
        """
        For simulation, no inverse transform is necessary, since
        data already have the dimensionality they need.

        Args:
            X (numpy.array): Array with targets to apply transformation to.
            Y (numpy.array): Array with responses to apply transformation to.
        """
        self._iX, self._iY = X, Y
        return self

    def transform(self, x, y=None):
        """
        Apply transformation to a single element in target space.

        Args:
            x (object): Single target to apply transformation to.
            y (object): Single response to apply transformation to.
        """
        return [x], [y]



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
        self.transforms = []
        for arg in args:
            self.add(arg)
        return

    def clone(self):
        """
        Clone object.
        """
        return self.__class__([xform.clone() for xform in self.transforms])

    def add(self, transform):
        """
        Add transform to internal transform list.
        """
        if not isinstance(transform, Transform):
            raise AssertionError('No rule for transforming data with {} object!'.format(str(type(transform))))
        self.transforms.append(transform)
        return

    def fit(self, X, Y=None, pred=False):
        """
        Traverse data and apply individual transformations. During
        this process, if the transformation is a simulator and the
        fit is being applied for prediction, skip that transform.

        Args:
            X (numpy.array): Array with targets to apply transformation to.
            Y (numpy.array): Array with responses to apply transformation to.
            pred (bool): Whether or not the transform is being applied
                for downstream prediction. If that's the case, simulators will
                be skipped.
        """
        # load from cache (if specified)
        if self.cache:
            data = self.load(X, Y)
            if data is not None:
                self._X, self._Y = data
                return

        # fit individual transforms
        tx, ty = X, Y
        for xf in self.transforms:
            if not (pred and isinstance(xf, (Simulator, ComplexSimulator))):
                tx, ty = xf.fit_transform(tx, ty)
        self._X, self._Y = tx, ty

        # save to cache (if specified)
        if self.cache:
            self.save(X, Y)
        return self

    def fit_transform(self, X, Y=None, pred=False):
        """
        Fit and transform full input target space.

        Args:
            X (numpy.array): Array with targets to apply transformation to.
            Y (numpy.array): Array with responses to apply transformation to.
            pred (bool): Whether or not the transform is being applied
                for downstream prediction. If that's the case, simulators will
                be skipped.
        """
        self.fit(X, Y, pred=pred)
        return self._X, self._Y

    def inverse_fit(self, X, Y=None):
        """

        """
        # inverse fit individual transforms
        tx, ty = X, Y
        for xf in self.transforms:
            tx, ty = xf.inverse_fit_transform(tx, ty)
        self._iX, self._iY = tx, ty
        return self
