# -*- coding: utf-8 -*-
#
# testing for main entry point
# 
# @author <bprinty@asuragen.com>
# ------------------------------------------------


# imporobj
# -------
import numpy

from jade import Transform, ComplexTransform, Simulator, ComplexSimulator, CompositeTransform


# transforms
# ----------
class SignalGenerator(Transform):
    funcs = {
        'sin': numpy.sin,
        'cos': numpy.cos
    }
    
    def __init__(self, period=1, samples=1000):
        self.period = period
        self.samples = samples
        return

    def transform(self, x, y=None):
        tx = self.funcs[x](2 * numpy.pi * self.period * numpy.arange(self.samples) / self.samples)
        return tx, y 


class VariableSignalGenerator(Transform):
    funcs = {
        'sin': numpy.sin,
        'cos': numpy.cos
    }
    
    def __init__(self, samples=1000):
        self.samples = samples
        return

    def transform(self, x, y=None):
        key = list(x.keys())[0]
        freq = x[key]
        tx = self.funcs[key](2 * numpy.pi * freq * numpy.arange(self.samples) / self.samples)
        return tx, y


class SegmentSignal(Transform):

    def __init__(self, chunksize=100):
        self.chunksize = chunksize
        return

    def transform(self, x, y=None):
        tx = numpy.array([x[i:(i+self.chunksize)] for i in range(0, len(x), self.chunksize)])
        if y is not None:
            y = numpy.array([y] * len(tx))
        return tx, y

    def inverse_transform(self, x, y=None):
        tx, ty = [], None
        for idx, arr in enumerate(x):
            tx.extend(x[idx])
        if y is not None:
            ty = numpy.median(y) == 1
        return tx, ty


class WhiteNoise(Simulator):

    def __init__(self, mu=0, sigma=0.01, clones=2):
        self.mu = mu
        self.sigma = sigma
        self.clones = clones
        return

    def transform(self, x, y=None):
        ty = [y] * (self.clones + 1) if y is not None else y
        tx = [x] + [
            numpy.array(x) + numpy.random.normal(self.mu, self.sigma, size=len(x))
            for i in range(0, self.clones)
        ]
        return tx, ty
