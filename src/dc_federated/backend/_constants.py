"""
Some constants to be used by classes in dc_federated.
"""

REGISTER_WORKER_ROUTE = 'register_worker'
RETURN_GLOBAL_MODEL_ROUTE = 'return_global_model'
QUERY_GLOBAL_MODEL_STATUS_ROUTE = 'query_global_model_status'
RECEIVE_WORKER_UPDATE_ROUTE = 'receive_worker_update'
WORKERS_ROUTE = 'workers'

WORKER_ID_KEY = 'worker_id'
MODEL_UPDATE_KEY = 'model_update'
ID_AND_MODEL_KEY = 'id_and_model'

WORKER_AUTHENTICATION_PHRASE = b'Please authenticate me'
NO_AUTHENTICATION = 'No Authentication'
AUTHENTICATED = 'Authenticated'
INVALID_WORKER = "Invalid Worker"
UNREGISTERED_WORKER = 'Unregistered Worker'

PUBLIC_KEY_STR = 'public_key_str'
SIGNED_PHRASE = 'signed_phrase'

ACTIVE_WORKER_KEY = 'active'

ADMIN_PASSWORD = 'DCF_SERVER_ADMIN_PASSWORD'
ADMIN_USERNAME = 'DCF_SERVER_ADMIN_USERNAME'

ERROR_MESSAGE_KEY = 'error'
SUCCESS_MESSAGE_KEY = 'success'
