# -*- coding: utf-8 -*-
#
# testing for main entry point
#
# ------------------------------------------------


# imporobj
# -------
import os
import unittest
import numpy
from sklearn.svm import SVC

from iso import Transform, ComplexTransform, TransformChain, Reduce
from iso import FeatureTransform
from .utils import tmpfile
from .utils import SignalGenerator
from .utils import VariableSignalGenerator
from .utils import SegmentSignal
from .utils import WhiteNoise
from .utils import NormalizedPower, DominantFrequency


# tests
# -----
class TestFeatureTransform(unittest.TestCase):
    # generate data:
    # here we're tyring to predict whether or not a
    # signal is above a periodicity of 5
    truth = [False] * 20 + [True] * 20
    data = [{'sin': i} for i in numpy.linspace(5, 10, 10)] + \
           [{'cos': i} for i in numpy.linspace(5, 10, 10)] + \
           [{'sin': i} for i in numpy.linspace(11, 15, 10)] + \
           [{'cos': i} for i in numpy.linspace(11, 15, 10)]

    def test_simple(self):
        xform = TransformChain(
            VariableSignalGenerator(fs=1000),
            FeatureTransform(
                NormalizedPower(),
                DominantFrequency(fs=1000)
            )
        )
        xform.fit(self.data[:10])
        self.assertEqual(
            list(map(list, numpy.round(xform._X, 2))),
            [[0.64, 5], [0.63, 6], [0.63, 6], [0.63, 7], [0.63, 7], [0.64, 8], [0.64, 8], [0.64, 9], [0.64, 9], [0.64, 10]]
        )
        self.assertEqual(len(xform._X), 10)
        return

    def test_chained(self):
        xform = TransformChain(
            VariableSignalGenerator(fs=1000),
            SegmentSignal(chunksize=200),
            Reduce(),
            FeatureTransform(
                NormalizedPower(),
                DominantFrequency(fs=1000)
            )
        )
        xform.fit(self.data)
        self.assertEqual([round(x, 2) for x in xform._X[0]], [0.64, 5])
        self.assertEqual([round(x, 2) for x in xform._X[-1]], [0.64, 15])
        self.assertEqual(len(xform._X), len(self.data) * 5)
        return
