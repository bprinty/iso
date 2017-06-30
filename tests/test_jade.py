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
import subprocess

import jade
from . import __base__, __resources__, tmpfile


# session
# -------
class TestEntryPoints(unittest.TestCase):

    def call(self, subcommand, *args):
        return subprocess.check_output('python -m jade {} {}'.format(
            subcommand, ' '.join(args)
        ), stderr=subprocess.STDOUT, shell=True, cwd=__base__)

    def test_version(self):
        res = self.call('version')
        self.assertTrue(res, jade.__version__)
        return

    def test_extract(self):
        return

    def test_fit(self):
        return

    def test_predict(self):
        return
