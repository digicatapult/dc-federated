"""
Some common utility functions.
"""
from dc_federated.backend._constants import GLOBAL_MODEL, GLOBAL_MODEL_VERSION


def create_model_dict(model_serialized, model_version):
    """
    Simple utility function to create the model dictionary as
    per the specification in the DCFServer.
    
    Parameters
    ----------
    
    model_serialized: byte-stream
        Serialized version of the model.

    model_version: object
        Version of the model

    Returns
    -------
    
    dict: 
        Dictionary with keys:
        GLOBAL_MODEL: containing the serialization of the global model
        GLOBAL_MODEL_VERSION: containing the global model itself.
    """
    return {
        GLOBAL_MODEL: model_serialized,
        GLOBAL_MODEL_VERSION: model_version
    }


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


def message_seriously_wrong(msg):
    return f"Something went seriously wrong - {msg}. Contact the application engineer immeidiately."


def verify_dict(dct, keys, data_types):
    """
    Verify that the dictionary has the necessary structure.

    Parameters
    ----------

    dct: object
        Object to verify as a dictionary

    keys: list
        The list of keys to verify

    data_types: list
        The list of values to verify as being of a specific type

    Returns
    -------
    list:
        list of keys where the validation failed.
    """
    if not isinstance(dct, dict):
        return keys
    return [key for key, vt in zip(keys, data_types)
            if key not in dct or not isinstance(dct[key], vt) ]








