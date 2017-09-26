# -*- coding: utf-8 -*-
#
# testing for main entry point
# 
# @author <bprinty@asuragen.com>
# ------------------------------------------------


# imporobj
# -------
import os
import unittest
import numpy
from sklearn.model_selection import cross_val_score
import pandas
import pytest
import random

from jade import Learner, Reduce, FeatureTransform
from . import __base__, __resources__, tmpfile
from .utils import VariableSignalGenerator, SegmentSignal, WhiteNoise
from .utils import NormalizedPower, DominantFrequency


# config
# ------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# tests
# -----
class TestLearner(unittest.TestCase):
    # generate data:
    # here we're tyring to predict whether or not a
    # signal is above a periodicity of 5
    truth = [False] * 20 + [True] * 20
    data = [{'sin': i} for i in numpy.linspace(5, 10, 10)] + \
           [{'cos': i} for i in numpy.linspace(5, 10, 10)] + \
           [{'sin': i} for i in numpy.linspace(11, 15, 10)] + \
           [{'cos': i} for i in numpy.linspace(11, 15, 10)]

    def test_transform(self):
        learner = Learner(
            transform=[
                VariableSignalGenerator(),
                SegmentSignal()
            ]
        )
        X, Y = learner.transform(self.data, self.truth)
        self.assertEqual(list(numpy.round(X[0][0][:3], 4)), [0, 0.0314, 0.0628])
        self.assertEqual(list(numpy.round(X[1][1][:3], 4)), [-0.342, -0.3746, -0.4067])
        self.assertEqual(list(Y[0][:3]), [0, 0, 0])
        self.assertEqual(list(Y[-1][:3]), [1, 1, 1])
        return

    def test_transform_features(self):
        learner = Learner(
            transform=[
                VariableSignalGenerator(fs=1000),
                SegmentSignal(chunksize=200),
                Reduce(),
                FeatureTransform(
                    NormalizedPower(),
                    DominantFrequency(fs=1000)
                )
            ]
        )
        X, Y = learner.transform(self.data, self.truth)
        self.assertEqual(len(Y), 200)
        self.assertEqual(len(X), 200)
        self.assertEqual(len(X[0]), 2)
        df = pandas.DataFrame(X, columns=learner.feature_names)
        self.assertEqual(len(df), 200)
        self.assertEqual(list(df.columns), ['NormalizedPower', 'DominantFrequency'])
        return

    def test_fit(self):
        learner = Learner(
            transform=[
                VariableSignalGenerator(),
                SegmentSignal()
            ]
        )
        learner.fit(self.data, self.truth)
        pred = learner.model.predict(numpy.matrix([
            learner.vectorizer._X[0][0],
            learner.vectorizer._X[1][0],
            learner.vectorizer._X[-2][0],
            learner.vectorizer._X[-1][0]
        ]))
        self.assertEqual(list(pred), [False, False, True, True])
        return

    def test_fit_with_simulator(self):
        learner = Learner(
            transform=[
                VariableSignalGenerator(),
                WhiteNoise(),
                SegmentSignal()
            ]
        )
        learner.fit(self.data, self.truth)
        pred = learner.model.predict(numpy.matrix([
            learner.vectorizer._X[0][0],
            learner.vectorizer._X[1][0],
            learner.vectorizer._X[-2][0],
            learner.vectorizer._X[-1][0]
        ]))
        self.assertEqual(list(pred), [False, False, True, True])
        return

    def test_fit_predict(self):
        # composite transform
        learner = Learner(
            transform=[
                VariableSignalGenerator(),
                SegmentSignal()
            ]
        )
        pred = learner.fit_predict(self.data, self.truth)
        self.assertEqual(list(pred), self.truth)

        # non-composite transform
        learner = Learner(
            transform=VariableSignalGenerator()
        )
        pred = learner.fit_predict(self.data, self.truth)
        self.assertEqual(list(pred), self.truth)
        return

    def test_predict(self):
        learner = Learner(
            transform=[
                VariableSignalGenerator(fs=1000),
                SegmentSignal(chunksize=20)
            ]
        )
        learner.fit(self.data, self.truth)
        pred = learner.predict([{'sin': 2}])
        self.assertEqual(pred[0], False)
        pred = learner.predict([{'sin': 12}])
        self.assertEqual(pred[0], True)
        
        # try it out with a response vector bigger
        # than the training response, to see if internal
        # transforms and inverse transforms are applied correctly
        data = [self.data[i] for i in range(0, len(self.data)) if i % 2]
        truth = [self.truth[i] for i in range(0, len(self.truth)) if i % 2]
        learner.fit(data, truth)
        pred = learner.predict(self.data)
        self.assertEqual(len(pred), len(self.data))
        self.assertEqual(pred[0], False)
        self.assertEqual(pred[-1], True)
        return

    def test_predict_with_simulator(self):
        learner = Learner(
            transform=[
                VariableSignalGenerator(),
                WhiteNoise(clones=2),
                SegmentSignal(),
                Reduce()
            ]
        )
        learner.fit(self.data, self.truth)
        pred = learner.predict([{'sin': 2}])
        self.assertEqual(pred[0], False)
        self.assertEqual(len(pred), 1)
        pred = learner.predict([{'sin': 12}, {'sin': 13}])
        self.assertEqual(pred[0], True)
        self.assertEqual(len(pred), 2)
        return


class TestModelPersistence(unittest.TestCase):
    # generate data:
    # here we're tyring to predict whether or not a
    # signal is above a periodicity of 5
    truth = [False] * 20 + [True] * 20
    data = [{'sin': i} for i in numpy.linspace(5, 10, 10)] + \
           [{'cos': i} for i in numpy.linspace(5, 10, 10)] + \
           [{'sin': i} for i in numpy.linspace(11, 15, 10)] + \
           [{'cos': i} for i in numpy.linspace(11, 15, 10)]

    def test_save(self):
        from sklearn.svm import SVC
        learner = Learner(
            transform=[
                VariableSignalGenerator(),
                WhiteNoise(clones=2),
                SegmentSignal(),
                Reduce()
            ], model=SVC(kernel='rbf')
        )
        learner.fit(self.data, self.truth)
        test = list(self.data)
        random.shuffle(test)
        pred = learner.predict(test)
        tmp = tmpfile('.pkl')
        learner.save(tmp)
        del learner
        learner = Learner.load(tmp)
        self.assertEqual(list(learner.predict(test)), list(pred))
        return

    def test_keras_save(self):
        try:
            from keras.models import Sequential
            from keras.layers import Dense, Dropout, Flatten, Reshape
            from keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D
            from keras.wrappers.scikit_learn import KerasClassifier
        except ImportError:
            self.skipTest('Keras not installed on this system.')
        def build():
            cnn = Sequential([
                Reshape((1, 100, 1), input_shape=(100,)),
                Conv2D(64, (3, 1), padding="same", activation="relu"),
                MaxPooling2D(pool_size=(1, 2)),
                Flatten(),
                Dense(128, activation="relu"),
                Dropout(0.2),
                Dense(1, activation='sigmoid'),
            ])
            cnn.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
            return cnn
        learner = Learner(
            transform=[
                VariableSignalGenerator(),
                WhiteNoise(clones=2),
                SegmentSignal(),
                Reduce()
            ], model=KerasClassifier(build)
        )
        learner.fit(self.data, self.truth, verbose=0)
        test = list(self.data)
        random.shuffle(test)
        pred = learner.predict(test, verbose=0)
        tmp = tmpfile('.pkl')
        learner.save(tmp)
        del learner
        learner = Learner.load(tmp)
        self.assertEqual(list(learner.predict(test, verbose=0)), list(pred))
        return


class TestExtensions(unittest.TestCase):
    # gnerate data:
    # here we're tyring to predict whether or not a
    # signal is above a periodicity of 5
    truth = [False] * 20 + [True] * 20
    data = [{'sin': i} for i in numpy.linspace(5, 10, 10)] + \
           [{'cos': i} for i in numpy.linspace(5, 10, 10)] + \
           [{'sin': i} for i in numpy.linspace(11, 15, 10)] + \
           [{'cos': i} for i in numpy.linspace(11, 15, 10)]

    @pytest.mark.skip(reason="feature currently in development")
    def test_scikit_validation(self):
        learner = Learner(
            transform=[
                VariableSignalGenerator(),
                SegmentSignal()
            ]
        )
        return