class MyMessage:
    """
        message type definition
    """
    # message to neighbor
    MSG_TYPE_INIT = 1
    MSG_TYPE_SEND_MSG_TO_NEIGHBOR = 2
    MSG_TYPE_METRICS = 3

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"
