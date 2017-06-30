# -*- coding: utf-8 -*-

import os
import sys
import uuid

__tests__ = os.path.dirname(os.path.realpath(__file__))
__base__ = os.path.realpath(os.path.join(__tests__, '..'))
__resources__ = os.path.join(__tests__, 'resources')
sys.path = [__base__] + sys.path


def tmpfile(ext):
    return os.path.join(__resources__, 'tmp.{}{}'.format(uuid.uuid1(), ext))


def tearDown():
    # clean up temporary files
    for i in os.listdir(__resources__):
        if i[0:4] == 'tmp.':
            os.remove(os.path.join(__resources__, i))
    return
