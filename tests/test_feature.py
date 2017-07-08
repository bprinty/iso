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
from sklearn.svm import SVC

from jade import Transform, ComplexTransform, TransformChain, Flatten
from jade import FeatureTransform
from . import __base__, __resources__, tmpfile
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

    def test_transform(self):
        generator = TransformChain(
            VariableSignalGenerator(fs=10000),
            WhiteNoise(sigma=0.1, clones=2),
            SegmentSignal(chunksize=200),
            Flatten()
        )
        X, Y = generator.fit_transform([{'sin': 100}, {'cos': 150}])
        xform = FeatureTransform(
            NormalizedPower(),
            DominantFrequency(fs=10000)
        )
        X, Y = xform.fit_transform(X, Y)
        # print X, Y
        # print X
        return

    def test_chained(self):
        # learner = Learner(
        #     transforms=[
        #         SignalGenerator(),
        #         FeatureTransform(
        #             Power(),
        #             Frequency()
        #         )
        #     ],
        #     model=SVC()
        # )
        # X, Y = xform.transform(self.data, self.truth)
        return
