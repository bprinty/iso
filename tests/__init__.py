# -*- coding: utf-8 -*-

import os
import sys

TESTS = os.path.dirname(os.path.realpath(__file__))
BASE = os.path.realpath(os.path.join(TESTS, '..'))
RESOURCES = os.path.join(TESTS, 'resources')
sys.path = [BASE] + sys.path
