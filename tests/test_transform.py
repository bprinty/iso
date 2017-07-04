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

from jade import Transform, ComplexTransform, CompositeTransform
from . import __base__, __resources__, tmpfile
from .utils import SignalGenerator
from .utils import VariableSignalGenerator
from .utils import SegmentSignal
from .utils import WhiteNoise


# tests
# -----
class TestTransform(unittest.TestCase):

    def test_hash(self):
        og = SignalGenerator(10, 1000)
        same = SignalGenerator(10, 1000)
        diff = SignalGenerator(5, 1000)
        self.assertEqual(len(og.hash(['sin', 'cos'])), 32)
        self.assertEqual(og.hash(['sin', 'cos']), same.hash(['sin', 'cos']))
        self.assertNotEqual(og.hash(['sin', 'cos']), same.hash(['sin', 'sin']))
        self.assertNotEqual(og.hash(['sin', 'cos']), diff.hash(['sin', 'cos']))
        return

    def test_caching(self):
        pkl = tmpfile('.pkl')

        # save
        xform = SignalGenerator()
        xform.fit(['sin', 'cos'])
        xform.save(['sin', 'cos'], filename=pkl)
        self.assertEqual(list(numpy.round(xform.X[0][:3], 4)), [0, 0.0063, 0.0126])
        self.assertEqual(list(numpy.round(xform.X[1][:3], 4)), [1.0, 1.0, 0.9999])

        # load
        xform._X, xform._Y = None, None
        X, Y = xform.load(['sin', 'cos'], filename=pkl)
        self.assertEqual(xform.X, None)
        self.assertEqual(xform.Y, None)
        self.assertEqual(list(numpy.round(X[0][:3], 4)), [0, 0.0063, 0.0126])
        self.assertEqual(list(numpy.round(X[1][:3], 4)), [1.0, 1.0, 0.9999])
        return

    def test_fit(self):
        xform = SignalGenerator()
        xform.fit(['sin', 'cos'])
        self.assertEqual(list(numpy.round(xform.X[0][:3], 4)), [0, 0.0063, 0.0126])
        self.assertEqual(list(numpy.round(xform.X[1][:3], 4)), [1.0, 1.0, 0.9999])
        return

    def test_fit_transform(self):
        xform = SignalGenerator()
        X, Y = xform.fit_transform(['sin', 'cos'])
        self.assertEqual(list(numpy.round(xform.X[0][:3], 4)), [0, 0.0063, 0.0126])
        self.assertEqual(list(numpy.round(xform.X[1][:3], 4)), [1.0, 1.0, 0.9999])
        self.assertEqual(list(xform.X[0][:10]), list(X[0][:10]))
        self.assertEqual(list(xform.X[1][:10]), list(X[1][:10]))
        return

    def test_fit_truth(self):
        xform = SegmentSignal(chunksize=5)
        xform.fit(numpy.array([
            numpy.array(range(0, 100)),
            numpy.array(range(0, 100))
        ]), [0, 1])
        self.assertEqual(list(xform.X[0][0][:3]), [0, 1, 2])
        self.assertEqual(list(xform.X[1][1][:3]), [5, 6, 7])
        self.assertEqual(list(xform.Y[0][:3]), [0, 0, 0])
        self.assertEqual(list(xform.Y[1][:3]), [1, 1, 1])
        return



class TestComplexTransform(unittest.TestCase):

    def test_fit(self):
        return

    def test_fit_transform(self):
        return


class TestSimulator(unittest.TestCase):

    def test_fit(self):
        gen = SignalGenerator()
        gen.fit(['sin', 'cos'])
        xform = WhiteNoise(clones=2)
        xform.fit(gen.X, gen.Y)
        self.assertEqual(list(numpy.round(xform.X[0][:3], 4)), [0, 0.0063, 0.0126])
        self.assertEqual(len(xform.X), 6)
        self.assertEqual(len(xform.X[0]), 1000)
        self.assertNotEqual(list(numpy.round(xform.X[1][:3], 4)), [1.0, 1.0, 0.9999])
        return


class TestComplexSimulator(unittest.TestCase):
    
    def test_fit(self):
        return


class TestCompositeTransform(unittest.TestCase):

    def test_fit(self):
        xform = CompositeTransform([
            SignalGenerator(),
            SegmentSignal()
        ])
        xform.fit(['sin', 'cos'])
        self.assertEqual(list(numpy.round(xform.X[0][0][:3], 4)), [0, 0.0063, 0.0126])
        self.assertEqual(list(numpy.round(xform.X[0][1][:3], 4)), [0.5878, 0.5929, 0.5979])
        self.assertEqual(len(xform.X[0][0]), xform.transforms[1].chunksize)
        self.assertEqual(len(xform.X[0][1]), xform.transforms[1].chunksize)
        self.assertEqual(list(numpy.round(xform.X[1][0][:3], 4)), [1.0, 1.0, 0.9999])
        self.assertEqual(list(numpy.round(xform.X[1][1][:3], 4)), [0.809, 0.8053, 0.8016])
        self.assertEqual(len(xform.X[1][0]), xform.transforms[1].chunksize)
        self.assertEqual(len(xform.X[1][1]), xform.transforms[1].chunksize)
        return

    def test_fit_transform(self):
        xform = CompositeTransform([
            SignalGenerator(),
            SegmentSignal()
        ])
        X, Y = xform.fit_transform(['sin', 'cos'])
        self.assertEqual(list(numpy.round(xform.X[0][0][:3], 4)), [0, 0.0063, 0.0126])
        self.assertEqual(list(numpy.round(xform.X[0][1][:3], 4)), [0.5878, 0.5929, 0.5979])
        self.assertEqual(len(xform.X[0][0]), xform.transforms[1].chunksize)
        self.assertEqual(len(xform.X[0][1]), xform.transforms[1].chunksize)
        self.assertEqual(list(numpy.round(xform.X[1][0][:3], 4)), [1.0, 1.0, 0.9999])
        self.assertEqual(list(numpy.round(xform.X[1][1][:3], 4)), [0.809, 0.8053, 0.8016])
        self.assertEqual(len(xform.X[1][0]), xform.transforms[1].chunksize)
        self.assertEqual(len(xform.X[1][1]), xform.transforms[1].chunksize)
        self.assertEqual(list(xform.X[0][0]), list(X[0][0]))
        self.assertEqual(xform.Y, Y)
        return

    def test_fit_simulate(self):
        xform = CompositeTransform([
            SignalGenerator(),
            WhiteNoise(clones=2),
            SegmentSignal()
        ])
        xform.fit(['sin', 'cos'])
        self.assertEqual(list(numpy.round(xform.X[0][0][:3], 4)), [0, 0.0063, 0.0126])
        self.assertNotEqual(list(numpy.round(xform.X[1][0][:3], 4)), [1.0, 1.0, 0.9999])
        self.assertEqual(len(xform.X[0][0]), xform.transforms[2].chunksize)
        self.assertEqual(len(xform.X[1][0]), xform.transforms[2].chunksize)
        return