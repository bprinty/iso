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

from jade import Learner, Flatten, FeatureTransform
from . import __base__, __resources__, tmpfile
from .utils import VariableSignalGenerator, SegmentSignal, WhiteNoise
from .utils import NormalizedPower, DominantFrequency


# tests
# -----
class TestLearn(unittest.TestCase):
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
                Flatten()
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
