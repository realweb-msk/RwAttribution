class Error(Exception):
    """Basic class to inherit from"""
    pass


class MissInputData(Error):
    """Raised when some expected input is missing"""
    pass


class NonListedValue(Error):
    """Raised when function get incorrect option"""
    pass
