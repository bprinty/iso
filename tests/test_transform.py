# -*- coding: utf-8 -*-
#
# testing for main entry point
#
# ------------------------------------------------


# imporobj
# -------
import unittest
import numpy

from iso import Transform, ComplexTransform, TransformChain, SimulatorGroup, Reduce
from .utils import tmpfile
from .utils import SignalGenerator
from .utils import VariableSignalGenerator
from .utils import SegmentSignal
from .utils import WhiteNoise
from .utils import FilterNoisiest


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


class TestReduce(unittest.TestCase):

    def test_already_flat(self):
        gen = TransformChain(
            VariableSignalGenerator(fs=10000),
            Reduce()
        )
        X, Y = gen.fit_transform([{'sin': 100}, {'cos': 150}], [0, 1])
        self.assertEqual(len(X), 2)
        self.assertEqual(len(X[0]), 10000)
        self.assertEqual(list(Y), [0, 1])
        return

    def test_fit_transform(self):
        # normal multi-layer transform
        gen = TransformChain(
            VariableSignalGenerator(fs=10000),
            WhiteNoise(sigma=0.1, clones=2),
            SegmentSignal(chunksize=200)
        )
        X, Y = gen.fit_transform([{'sin': 100}, {'cos': 150}], [0, 1])
        self.assertEqual(len(X), len(Y))
        self.assertEqual(len(X), 6)
        self.assertEqual(len(X[0]), len(Y[0]))
        self.assertEqual(len(X[0]), 50)
        self.assertEqual(len(X[0][0]), 200)

        # add flattening layer
        gen.add(Reduce())
        X, Y = gen.fit_transform([{'sin': 100}, {'cos': 150}], [0, 1])
        self.assertEqual(len(X), len(Y))
        self.assertEqual(len(X), 300)
        self.assertEqual(len(X[0]), 200)
        return

    def test_inverse_fit_transform(self):
        # non-flattening transform
        ogen = TransformChain(
            VariableSignalGenerator(fs=10000),
            WhiteNoise(sigma=0.1, clones=2),
            SegmentSignal(chunksize=200)
        )
        
        # flattening transform
        fgen = ogen.clone()
        fgen.add(Reduce())
        
        # transform/inverse-transform
        fX, fY = fgen.fit_transform([{'sin': 100}, {'cos': 150}], [0, 1])
        tX, tY = fgen.inverse_fit_transform(fX, fY)
        
        # expected inversion
        fX, fY = ogen.fit_transform([{'sin': 100}, {'cos': 150}], [0, 1])
        oX, oY = ogen.inverse_fit_transform(fX, fY)
        
        # check dimensionality
        self.assertEqual(len(oX), len(tX))
        self.assertEqual(len(oX[0]), len(tX[0]))
        self.assertEqual(len(oY), len(tY))
        self.assertEqual(list(oX[3]), list(tX[3]))
        self.assertEqual(list(oY), list(tY))
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
        gen = SignalGenerator()
        gen.fit(['sin', 'cos'])
        xform = TransformChain([
            WhiteNoise(mu=0, sigma=2, clones=10),
            FilterNoisiest(),
        ])
        xform.fit(gen.X, gen.Y)
        self.assertEqual(len(xform.X), 20)
        self.assertEqual(len(xform.transforms[0].X), 22)
        self.assertEqual(len(xform.transforms[1].X), 20)
        return


class TestSimulatorGroup(unittest.TestCase):
    
    def test_fit(self):
        gen = SignalGenerator()
        gen.fit(['sin', 'cos'])
        xform = SimulatorGroup(
            WhiteNoise(mu=10, sigma=5, clones=2),
            WhiteNoise(mu=0, sigma=0.1, clones=2)
        )
        xform.fit(gen.X, gen.Y)
        self.assertEqual(list(numpy.round(xform.X[0][:3], 4)), [0, 0.0063, 0.0126])
        self.assertEqual(list(numpy.round(xform.X[1][:3], 4)), [12.4836, 9.315, 13.251])
        self.assertEqual(list(numpy.round(xform.X[-2][:3], 4)), [0.8886, 0.9369, 0.9057])
        self.assertEqual(list(numpy.round(xform.X[-1][:3], 4)), [1.0785, 0.8222, 1.0714])
        self.assertEqual(len(xform.X), 12)
        self.assertEqual(len(xform.X[0]), 1000)
        return


class TestTransformChain(unittest.TestCase):

    def test_fit(self):
        xform = TransformChain([
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
        xform = TransformChain([
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
        xform = TransformChain([
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

    def test_transform(self):
        xform = TransformChain([
            SignalGenerator(),
            SegmentSignal()
        ])
        x, y = xform.transform('sin')
        self.assertEqual(list(numpy.round(x[0][:3], 4)), [0, 0.0063, 0.0126])
        self.assertEqual(len(x[0]), xform.transforms[1].chunksize)
        self.assertEqual(len(x[1]), xform.transforms[1].chunksize)
        return
