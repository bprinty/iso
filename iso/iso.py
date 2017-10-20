# -*- coding: utf-8 -*-
#
# Core configuration
# 
# @author <bprinty@gmail.com>
# ------------------------------------------------


# imports
# -------
import os
from gems import composite

from . import exc


# config
# ------
__base__ = os.path.dirname(os.path.realpath(__file__))
__default_config__ = os.path.join(__base__, '.iso')
__user_config__ = os.path.join(os.getenv("HOME"), '.iso')


class Session(object):
    """
    Object to manage all session configuration.
    """
    
    def __init__(self, cpus=1, db='/data/iso', data=None, models=None):
        self.cpus = cpus
        self.db = db
        self.models = os.path.join(self.db, 'models') if models is None else models
        self.data = os.path.join(self.db, 'data') if data is None else data
        return


def options(**kwargs):
    """
    Set options for the current session.

    Args:
        kwargs (dict): List of arbitrary config items to set.
    """
    global session, __default_config__, __user_config__

    # read default config
    with open(__default_config__, 'r') as cfig:
        config = composite.load(cfig)
    
    # read and add user config
    if os.path.exists(__user_config__):
        with open(__user_config__, 'r') as cfig:
            config = config + composite.load(cfig)
    if len(kwargs) != 0:
        config = config + composite(kwargs)

    # update session
    try:
        session = Session(**config._dict)
    except TypeError:
        raise exc.SettingsError()
    return config.json()


session = Session()
options()
