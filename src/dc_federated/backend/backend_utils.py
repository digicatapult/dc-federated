"""
Some common utility functions.
"""
from dc_federated.backend._constants import GLOBAL_MODEL, GLOBAL_MODEL_VERSION


def is_valid_model_dict(data):
    """
    Checks if the argument passed is a valid model dictionary.

    Parameters
    ----------

    data: object
        The object to check for validity

    Returns
    -------

    bool:
        Whether the object passes the test
    """
    return (isinstance(data, dict) and
        GLOBAL_MODEL in data and
        GLOBAL_MODEL_VERSION in data)
