# -*- coding: utf-8 -*-

# config
# ------
__author__ = 'Blake Printy'
__email__ = 'bprinty@asuragen.com'
__version__ = '0.0.1'


# imports
# -------
from .jade import options

from .feature import Feature
from .feature import ComplexFeature
from .feature import FeatureTransform

from .transform import Transform
from .transform import ComplexTransform
from .transform import TransformChain

from .transform import Flatten

from .transform import Simulator
from .transform import ComplexSimulator

from .learn import Learner


# [
#     [1, 2, 3, ..., 1000],
#     [1, 2, 3, ..., 1000],
# ]

# [
#     [
#         [1, 2, 3]
#         [4, 5, 6]
#     ]
#     [
#         [1, 2, 3]
#         [4, 5, 6]
#     ]
# ]
# [
#     [1, 2, 3],
#     [4, 5, 6],
#     [1, 2, 3],
#     [4, 5, 6]
# ]
