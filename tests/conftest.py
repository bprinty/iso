# -*- coding: utf-8 -*-
#
# pytest plugins
# 
# @author <bprinty@gmail.com>
# ------------------------------------------------


# imports
# -------
import pytest
from . import tearDown


# plugins
# -------
def pytest_addoption(parser):
    parser.addoption("-E", action="store", metavar="NAME",
        help="only run tests matching the environment NAME.")
    return


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line("markers",
        "env(name): mark test to run only on named environment")
    return


def pytest_runtest_setup(item):
    envmarker = item.get_marker("env")
    if envmarker is not None:
        envname = envmarker.args[0]
        if envname != item.config.getoption("-E"):
            pytest.skip("test requires env %r" % envname)
    return


@pytest.fixture(autouse=True)
def remove_tmpfiles():
    yield
    tearDown()
    return
