# -*- coding: utf-8 -*-
#
# custom exceptions
# 
# @author <bprinty@gmail.com>
# ------------------------------------------------


# imports
# -------
# none yet ...

# exc
# ---
class MissingResourceError(Exception):
    """
    Exception for handling missing resources.

    Args:
        path (str): Path to missing resource.
    """

    def __init__(self, path):
        self.path = path
        self.message = ('Could not access resource {}. The file '
                        'is either missing or malformed.'.format(self.path))
        return


class SettingsError(Exception):
    """
    Exception for handling missing resources.

    Args:
        data (dict): Settings data that was invalid.
    """

    def __init__(self, data):
        self.data = data
        self.message = ('Setting data {} is unsupported. Please see the '
                        'documentation for the list of available settings.'.format(str(self.data)))
        return
