"""
  module defines exceptions for cosmopy
  
"""

class Error(Exception):
    """base class for cosmopy exceptions"""
    pass

class UnknownParameterError(Error):
    """
    exception raised when a parameter is not in the default dict
    """
    pass

class ZNotZeroInBondEfsTk(Error):
    """
    exception raised when the experimental Bond and Efsthatiou T(k)
    is called at z != 0
    """
    pass

