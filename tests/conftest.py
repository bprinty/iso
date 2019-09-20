# -*- coding: utf-8 -*-
#
# pytest plugins
# 
# @author <bprinty@gmail.com>
# ------------------------------------------------


# imports
# -------
import os
import pytest
from . import RESOURCES


# plugins
# -------
@pytest.fixture(autouse=True)
def cleanup():

    yield

    for i in os.listdir(RESOURCES):
        if i[0:4] == 'tmp.':
            os.remove(os.path.join(RESOURCES, i))
    return
