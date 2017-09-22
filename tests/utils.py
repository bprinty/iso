# -*- coding: utf-8 -*-
#
# testing for main entry point
# 
# @author <bprinty@asuragen.com>
# ------------------------------------------------


# imporobj
# -------
import numpy

from jade import Transform, ComplexTransform, TransformChain
from jade import Simulator, ComplexSimulator
from jade import Feature, FeatureTransform


# transforms
# ----------
class SignalGenerator(Transform):
    funcs = {
        'sin': numpy.sin,
        'cos': numpy.cos
    }
    
    def __init__(self, f=1, fs=1000):
        self.f = f
        self.fs = fs
        return

    def transform(self, x, y=None):
        t = numpy.arange(self.fs) / float(self.fs)
        tx = self.funcs[x](2 * numpy.pi * self.f * t)
        return tx, y 


class VariableSignalGenerator(Transform):
    funcs = {
        'sin': numpy.sin,
        'cos': numpy.cos
    }
    
    def __init__(self, fs=1000):
        self.fs = fs
        return

    def transform(self, x, y=None):
        key = list(x.keys())[0]
        f = x[key]
        t = numpy.arange(self.fs) / float(self.fs)
        tx = self.funcs[key](2 * numpy.pi * f * t)
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

    def inverse_transform(self, w, x, y=None):
        if y is not None:
            y = numpy.median(y) > 0
        return w, y


class WhiteNoise(Simulator):

    def __init__(self, mu=0, sigma=0.01, clones=2):
        self.mu = mu
        self.sigma = sigma
        self.clones = clones
        numpy.random.seed(42)
        return

    def transform(self, x, y=None):
        ty = [y] * (self.clones + 1) if y is not None else y
        tx = [x] + [
            numpy.array(x) + numpy.random.normal(self.mu, self.sigma, size=len(x))
            for i in range(0, self.clones)
        ]
        return tx, ty


class NormalizedPower(Feature):
    
    def transform(self, x):
        return numpy.sum(numpy.abs(x)) / numpy.size(x)


class DominantFrequency(Feature):
    
    def __init__(self, fs=1000):
        self.fs = fs
        return
    
    def transform(self, x):
        fx = numpy.fft.fft(x)
        ft = numpy.fft.fftfreq(len(x), 1.0 / float(self.fs))
        pfreq = ft[numpy.where(ft >= 0)]
        pmag = abs(fx[numpy.where(ft >= 0)])
        idx = numpy.argmax(pmag)
        return pfreq[idx]
