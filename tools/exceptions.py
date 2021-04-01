class Error(Exception):
    """Basic class to inherit from"""
    pass


class MissInputData(Error):
    """Raised when some expected input is missing"""
    pass
