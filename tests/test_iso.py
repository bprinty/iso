# -*- coding: utf-8 -*-
#
# testing for main entry point
#
# ------------------------------------------------


# imporobj
# -------
import os
import sys
import unittest
import subprocess

import iso
from . import BASE


# session
# -------
class TestEntryPoints(unittest.TestCase):

    def call(self, subcommand, *args):
        python = 'python3' if sys.version_info > (3, 0) else 'python'
        return subprocess.check_output('{} -m iso {} {}'.format(
            python, subcommand, ' '.join(args)
        ), stderr=subprocess.STDOUT, shell=True, cwd=BASE, env=os.environ.copy())

    def test_version(self):
        res = self.call('version')
        self.assertTrue(res, iso.__version__)
        return

    def test_extract(self):
        # NO ENTRY POINTS YET ... THIS IS A PLACEHOLDER
        return

    def test_fit(self):
        # NO ENTRY POINTS YET ... THIS IS A PLACEHOLDER
        return

    def test_predict(self):
        # NO ENTRY POINTS YET ... THIS IS A PLACEHOLDER
        return
