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
from copy import copy, deepcopy
import hashlib
import logging

from .jade import session


# transformation
# --------------
class Transform(TransformerMixin):
    """
    Data transform operator, transforming data from one space
    into another space for downstream processing.
    """
    registered = True
    cache = False
    jobs = 1
    _X, _Y = None, None
    _iX, _iY = None, None

    def __add__(self, other):
        if isinstance(self, TransformChain):
            xf = self.clone().transforms
        else:
            xf = [self.clone()]
        if isinstance(other, TransformChain):
            return TransformChain(xf + other.clone().transforms)
        elif isinstance(other, Transform):
            return TransformChain(xf + [other.clone()])
        else:
            raise AssertionError('No rule for adding type {} and {}'.format(str(type(self)), str(type(other))))
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def clone(self):
        """
        Produce "copy" of transform that can be used fit data
        outside of the current context.
        """
        ret = copy(self)
        ret._X, ret._Y, ret._W = None, None, None
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
    
    def fit(self, X, Y=None, jobs=1):
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

        # register
        if not self.registered:
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
                # the index for Y needs to be capped, because when
                # vectorization is applied in the context of prediction
                # via learner, we can't guarantee that the predictors
                # and original truth are the same length
                tx, ty = self.transform(x, Y[min(ix, len(Y) - 1)])
                self.add_block('Y', ty)
            self.add_block('X', tx)
        self._X = numpy.array(self._X)
        self._W = X
        self._Y = numpy.array(self._Y) if Y is not None else Y
        
        # assert that predictor and response have
        # same outer dimensionality
        if Y is not None and len(self._X) != len(self._Y):
            raise AssertionError('Error: Target and response vectors must have the same outer dimensionality!')
        
        # save to cache (if specified)
        if self.cache:
            self.save(X, Y)
        return self

    def fit_transform(self, X, Y=None, jobs=1):
        """
        Fit and transform full input target space.

        Args:
            X (numpy.array): Array with targets to apply transformation to.
            Y (numpy.array): Array with responses to apply transformation to.
        """
        self.fit(X, Y, jobs=jobs)
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
                tx, ty = self.inverse_transform(self._W[min(ix, len(self._W) - 1)], x)
            else:
                tx, ty = self.inverse_transform(self._W[min(ix, len(self._W) - 1)], x, Y[min(ix, len(Y) - 1)])
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

    def inverse_transform(self, w, x, y=None):
        """
        Apply inverse transformation to a single element in target space.

        Args:
            idx (int): Index of data in original data structure.
            x (object): Single target to apply transformation to.
            y (object): Single response to apply transformation to.
        """
        return w, y

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


class ComplexTransform(Transform):
    """
    Complex data transform operator requiring two passes through the data - 
    once for a "peek" at the landscape of the data, and another for
    transforming the data into another space.
    """
    registered = False


# simulation
# ----------
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


class ComplexSimulator(Simulator):
    """
    Transform for simulating new data for model building in a way
    that requires a peak at the data before data are simulated. These
    transformations don't change the dimensionality of data, they
    just 
    """
    registered = False


class SimulatorGroup(Simulator):
    """
    Transform for combining multiple simulators in parallel,
    so that all simulators see the same input data and aren't
    required to be applied in a chain.
    """

    def __init__(self, *args):
        if len(args) != 0 and isinstance(args[0], (list, tuple)):
            args = args[0]
        self.simulators = []
        for arg in args:
            self.add(arg)
        return

    def __repr__(self):
        return '{}(\n\t{})'.format(self.__class__.__name__, ',\n\t'.join(map(repr, self.simulators)))

    def __getitem__(self, key):
        """
        Return transform with in chain. If the input is a slice,
        return a new SimulatorGroup with that slice of simulators.
        """
        if isinstance(key, slice):
            return SimulatorGroup(self.simulators[key])
        return self.simulators[key]

    def __iter__(self):
        """
        Generator for the object. Returns individual simulators during
        each phase of iteration.
        """
        for xf in self.simulators:
            yield xf
        return

    def clone(self):
        """
        Clone object.
        """
        return self.__class__([sim.clone() for sim in self.simulators])

    def add(self, simulator):
        """
        Add transform to internal transform list.
        """
        if not isinstance(simulator, Simulator):
            raise AssertionError('No rule for transforming data with {} object!'.format(str(type(simulator))))
        self.simulators.append(simulator)
        return

    @property
    def registered(self):
        """
        Proxy for checking if all internal simulators are registered.
        """
        for sim in self.simulators:
            if not sim.registered:
                return False
        return True

    def register(self, x, y=None):
        """
        Register a single element in target space.

        Args:
            x (object): Single target to apply transformation to.
            y (object): Single response to apply transformation to.
        """
        for sim in self.simulators:
            if self.registered:
                break
            sim.register(x, y)
        return

    def parameterize(self):
        """
        Parameterize models for downstream transformation.
        """
        for sim in self.simulators:
            sim.parameterize()
        return

    def transform(self, x, y=None):
        """
        Apply transformation to a single element in target space.

        Args:
            x (object): Single target to apply transformation to.
            y (object): Single response to apply transformation to.
        """
        ox, oy = [], []
        for sim in self.simulators:
            tx, ty = sim.transform(x, y)
            ox.extend(tx)
            if y is not None:
                oy.extend(ty)
        return ox, oy if y is not None else y


# flatteners
# ----------
class Reduce(Transform):
    """
    Transform for reducing dimensionality of input by specified
    number of dimensions. This used to be called "Flatten", but was
    changed to "Reduce" to be compatible with keras layers without
    collision.
    """

    def fit(self, X, Y=None, jobs=1):
        """
        "Flatten" input, changing dimensionality into
        something conducive to AI model development. In a nutshell,
        this decreases the dimensionality of predictors and responses
        until the response vector is one-dimensional.
        """
        fX, fY, fZ = [], [], []
        for ix in range(0, len(X)):
            
            # if the response vector is flat
            if Y is not None and not isinstance(Y[min(ix, len(Y) - 1)], (list, tuple, numpy.ndarray)):
                self._X, self._Y, self._Z = X, Y, None
                return self
            
            # the predictor vector is already flat
            elif not isinstance(X[ix], (list, tuple, numpy.ndarray)):
                self._X, self._Y, self._Z = X, Y, None
                return self
            
            # otherwise, flatten
            else:
                fX.extend(X[ix])
                fZ.append(len(X[ix]))
                if Y is not None:
                    fY.extend(Y[min(ix, len(Y) - 1)])
        
        # store the results
        self._X = numpy.array(fX)
        self._Y = numpy.array(fY) if Y is not None else Y
        self._Z = numpy.array(fZ)
        return self

    def inverse_fit(self, X, Y=None):
        """
        "Inverse flatten" input, changing dimensionality back into
        space that can be back-transformed into something
        human-interpretable.
        """
        if self._Z is None:
            self._X, self._Y = X, Y
            return self
        cidx = 0
        fX, fY = [], []
        for z in self._Z:
            fX.append(X[cidx:(cidx + z)])
            if Y is not None:
                fY.append(Y[cidx:(cidx + z)])
            cidx += z
        self._iX = numpy.array(fX)
        self._iY = numpy.array(fY) if Y is not None else Y
        return self


# vectorizers
# -----------
class TransformChain(Transform):
    """
    Manager for all transpose layers.
    """

    def __init__(self, *args):
        if len(args) != 0 and isinstance(args[0], (list, tuple)):
            args = args[0]
        self.transforms = []
        for arg in args:
            self.add(arg)
        return

    def __repr__(self):
        return '{}(\n\t{})'.format(self.__class__.__name__, ',\n\t'.join(map(repr, self.transforms)))

    def __getitem__(self, key):
        """
        Return transform with in chain. If the input is a slice,
        return a new TransformChain with that slice of transforms.
        """
        if isinstance(key, slice):
            return TransformChain(self.transforms[key])
        return self.transforms[key]

    def __iter__(self):
        """
        Generator for the object. Returns individual transforms during
        each phase of iteration.
        """
        for xf in self.transforms:
            yield xf
        return

    @property
    def has_simulator(self):
        """
        Return boolean describing whether or not chain has simulator.
        This is used in learner objects to predict on data without
        simulation for fit_predict methods.
        """
        for xf in self.transforms:
            if isinstance(xf, Simulator):
                return True
        return False

    @property
    def is_complex(self):
        """
        Return boolean describing whether or not chain has a complex
        operation within it.
        """
        for xf in self.transforms:
            if isinstance(xf, (ComplexTransform, ComplexSimulator)):
                return True
        return False

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

    def fit(self, X, Y=None, jobs=1, pred=False):
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
        self._W = X

        # save to cache (if specified)
        if self.cache:
            self.save(X, Y)
        return self

    def fit_transform(self, X, Y=None, jobs=1, pred=False):
        """
        Fit and transform full input target space.

        Args:
            X (numpy.array): Array with targets to apply transformation to.
            Y (numpy.array): Array with responses to apply transformation to.
            pred (bool): Whether or not the transform is being applied
                for downstream prediction. If that's the case, simulators will
                be skipped.
        """
        self.fit(X, Y, jobs=1, pred=pred)
        return self._X, self._Y

    def inverse_fit(self, X, Y=None, Z=None):
        """
        Inverse fit and transform targets back into input space.

        Args:
            X (numpy.array): Array with targets to apply transformation to.
            Y (numpy.array): Array with responses to apply transformation to.
            Z (numpy.array): Array with original targets transformation was applied to.
                These are often useful for back-transforming responses into the
                space they were originally supplied in.
        """
        # inverse fit reversed individual transforms
        tx, ty = X, Y
        for idx, xf in enumerate(self.transforms[::-1]):
            tx, ty = xf.inverse_fit_transform(tx, ty)
        self._iX, self._iY = tx, ty
        return self

    def transform(self, x, y=None):
        """
        Apply non-complex transformations to a single element in target space.
        This method isn't used for fitting models, but is included to
        do quick transformations on single data points for simple transforms.

        Args:
            x (object): Single target to apply transformation to.
            y (object): Single response to apply transformation to.
        """
        if self.is_complex:
            raise AssertionError('Cannot apply complex transform operation to '
                                 'single instance. This chain has a complex operation.')
        tx, ty = x, y
        for xf in self.transforms:
            tx, ty = xf.transform(tx, ty)
        return tx, ty
