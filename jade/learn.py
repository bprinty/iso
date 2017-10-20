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
import logging
from sklearn.base import BaseEstimator, clone
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, ShuffleSplit
from sklearn.exceptions import UndefinedMetricWarning
from sklearn import metrics
from gems import composite
from cached_property import cached_property

from .transform import TransformChain, Reduce
from .feature import FeatureTransform
from .jade import session


# config
# ------
__models__ = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models'))
warnings.simplefilter("ignore", UndefinedMetricWarning)


# functions
# ---------
def prepare_metric(metric):
    if isinstance(metric, dict):
        res = lambda x, y: {k: metric[k](x, y) for k in metric}
    elif isinstance(metric, (list, tuple)):
        res = lambda x, y: [func(x, y) for func in metric]
    else:
        res = metric
    return res


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
        self.filename = filename

        # handle non-compatible keras format
        if 'keras.models.Sequential' in str(type(model)):
            raise AssertionError('Please use KerasClassifier or KerasRegressor object with jade Learner! See https://keras.io/scikit-learn-api/ for more info.')

        # save into non-parsed property (run generator if available)
        # self._model is used here so that different types of models from
        # different libraries can be resolved properly
        if callable(model):
            self._model = model()
        else:
            self._model = model
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
        if 'Keras' in str(type(self.model)):
            model = {
                'config': self.model.model.get_config(),
                'weights': self.model.model.get_weights(),
                'type': self.model.__class__.__name__
            }
            if model['type'] == 'KerasClassifier':
                model['classes'] = self.model.classes_
                model['n_classes'] = self.model.n_classes_
        else:
            model = self.model
        joblib.dump(
            self.__class__(
                transform=self.vectorizer.clone(),
                model=model,
                shaper=self.shaper.clone()
            ), filename
        )
        return

    @cached_property
    def model(self):
        """
        Model component of Learner object, implemented as cached_property
        so that model persisitance can be achieved with all of the
        different types of models from different libraries.
        """
        # scikit-learn model definition
        if isinstance(self._model, BaseEstimator):
            model = self._model
        
        # keras classifier object
        elif 'KerasClassifier' in str(type(self._model)):
            model = self._model
        
        # keras model definition
        elif isinstance(self._model, dict) and 'config' in self._model:
            from keras.models import Sequential
            from keras.wrappers import scikit_learn
            cl = getattr(scikit_learn, self._model['type'])(lambda: 1)
            cl.model = Sequential.from_config(self._model['config'])
            cl.model.set_weights(self._model['weights'])
            if self._model['type'] == 'KerasClassifier':
                cl.classes_ = self._model['classes']
                cl.n_classes_ = self._model['n_classes']
            model = cl

        # pickle object
        elif isinstance(self._model, basestring):
            model = joblib.load(self._model)
        return model

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

    def transform(self, X, Y=None, jobs=1):
        """
        Transform input data into ai-ready tensor.
        """
        obj = self.vectorizer.clone()
        X, Y = obj.fit_transform(X, Y, jobs=jobs, pred=True)
        return X, Y

    def score(self, X, Y, jobs=1, metric=metrics.accuracy_score, **kwargs):
        """
        Apply model scoring function on transformed data.
        """
        metric = prepare_metric(metric)
        obj = self.vectorizer.clone()
        tX, tY = obj.fit_transform(X, Y, jobs=jobs, pred=True)
        fX, fY = self.flatten(tX, tY)
        pY = self.model.predict(fX, **kwargs)
        return metric(fY, pY, **kwargs)

    def fit(self, X, Y, jobs=1, **kwargs):
        """
        Train learner for speicific data indices.
        """
        self.shaper = None
        tX, tY = self.vectorizer.fit_transform(X, Y, jobs=jobs)
        self._X, self._Y = self.flatten(tX, tY)
        self.model.fit(self._X, self._Y, **kwargs)
        return self

    def fit_transform(self, X, Y, jobs=1, **kwargs):
        self.fit(X, Y, jobs=jobs, **kwargs)
        return self._X, self._Y

    def fit_predict(self, X, Y, jobs=1, **kwargs):
        """
        Fit models to data and return prediction.
        """
        self.fit(X, Y, jobs=jobs, **kwargs)
        if self.vectorizer.has_simulator:
            # we don't want to make predictions on the simulated
            # data, because it's only used to boost the training set
            return self.predict(X, jobs=jobs, **kwargs)
        else:
            # if we can fully back-transform the data, there's
            # no need to re-do the transformation process
            pY = self.model.predict(self._X, **kwargs)
            fX, fY = self.inverse_flatten(self._X, pY)
            rX, rY = self.vectorizer.inverse_fit_transform(fX, fY)
        return rY

    def predict(self, X, jobs=1, **kwargs):
        """
        Predict results from new data.
        """
        obj = self.vectorizer.clone()
        tX, tY = obj.fit_transform(X, None, jobs=jobs, pred=True)
        fX, fY = self.flatten(tX, self.vectorizer[-1]._Y if self.shaper is None else None)
        pY = self.model.predict(fX, **kwargs)
        fX, fY = self.inverse_flatten(fX, pY)
        rX, rY = obj.inverse_fit_transform(fX, fY)
        return rY


# model validation
# ----------------
class Validator(object):
    """
    Validation methods for testing out different models.
    """
    def __init__(self, learner, targets, truth):
        self.learner = learner
        self.targets = targets
        self.truth = truth
        return

    def cv_score(self, folds=5, schedule=None, stratified=False,
        seed=42, metric=metrics.accuracy_score,
        verbose=False, jobs=1):
        """
        Run cross-validation for learner using specified schedule
        or number of folds.
        """
        if schedule is None:
            if folds > len(self.targets):
                raise AssertionError("Number of folds must be less than size of data!")
            if folds < len(self.targets):
                if stratified:
                    # TODO: figure out how to do stratified split here
                    #       using the StratifiedKFold class. It would require
                    #       a transformation of the truth vector and a notion of
                    #       what events are available after transformation.
                    raise NotImplementedError('Stratified train/test splitting not yet implemented!')
                else:
                    schedule = KFold(n_splits=folds, random_state=seed)
            else:
                schedule = LeaveOneOut()
        scores = []
        for itrain, itest in schedule.split(self.targets):
            xtrain = [self.targets[i] for i in itrain]
            ytrain = [self.truth[i] for i in itrain]
            xtest = [self.targets[i] for i in itest]
            ytest = [self.truth[i] for i in itest]
            self.learner.fit(xtrain, ytrain, jobs=jobs)
            scores.append(self.learner.score(xtest, ytest, metric=metric, jobs=jobs))
        return scores

    def split_score(self,
        test_size=None, train_size=None,
        seed=42, metric=metrics.accuracy_score,
        verbose=False, jobs=1):
        """
        Run validation for learner using simple train/test split of
        data according to specified proportions.
        """
        xtrain, xtest, ytrain, ytest = train_test_split(
            self.targets, self.truth,
            test_size=test_size,
            train_size=train_size,
            random_state=seed
        )
        self.learner.fit(xtrain, ytrain, jobs=jobs)
        return self.learner.score(xtest, ytest, metric=metric, jobs=jobs)

    def bootstrap_score(self,
        splits=3, test_size=0.3, train_size=None,
        seed=42, metric=metrics.accuracy_score,
        verbose=False, jobs=1):
        """
        Run bootstrap validation for the data, randomly generating traning
        and testing sets of specified proportions for a number of splits.
        """
        return self.cv_score(
            schedule=ShuffleSplit(
                n_splits=splits,
                test_size=test_size,
                train_size=train_size,
                random_state=seed
            ), metric=metric,
            verbose=verbose
        )

    def identity_score(self,
        metric=metrics.accuracy_score,
        verbose=False, jobs=1):
        """
        Validate prediction performance of the dataset on itself.
        """
        self.learner.fit(self.targets, self.truth, jobs=jobs)
        return self.learner.score(
            self.targets, self.truth, metric=metric, jobs=jobs)

