import zmq
import pickle
import logging
import numpy as np

def setup_publisher(context, port):
    """
    Create and bind a publisher socket for sending model weights.
    """
    try:
        publisher_socket = context.socket(zmq.PUB)
        publisher_socket.bind(f"tcp://*:{port}")
        publisher_socket.setsockopt(zmq.SNDHWM, 1000)
        logging.info(f"Publisher bound to port {port}.")
        return publisher_socket
    except zmq.ZMQError as e:
        logging.error(f"Error setting up publisher on port {port}: {e}")
        raise

def setup_subscribers(context, ports):
    """
    Create subscriber sockets that listen to other nodes.
    """
    subscriber_sockets = []
    for port in ports:
        try:
            subscriber_socket = context.socket(zmq.SUB)
            subscriber_socket.connect(f"tcp://localhost:{port}")
            subscriber_socket.setsockopt(zmq.SUBSCRIBE, b"")
            subscriber_socket.setsockopt(zmq.RCVHWM, 1000)
            subscriber_socket.setsockopt(zmq.RCVBUF, 1024 * 1024 * 10)

            logging.info(f"Subscriber connected to port {port}.")
            subscriber_sockets.append(subscriber_socket)
        except zmq.ZMQError as e:
            logging.error(f"Error setting up subscriber on port {port}: {e}")
            raise
    return subscriber_sockets

def serialize_weights(weights):
    """Convert weights to bytes before sending."""
    return pickle.dumps(weights)

def deserialize_weights(serialized_weights):
    """Convert received bytes back to Python objects."""
    return pickle.loads(serialized_weights)

def decompress_gradients(compressed_grads):
    """
    Convert int8 weights back to float values.
    """
    return [grad / 127.0 for grad in compressed_grads]

def receive_nonblocking(subscriber_socket):
    """
    Try to receive a message without blocking the program.
    """
    try:
        message = subscriber_socket.recv(zmq.NOBLOCK)
        if message:
            return deserialize_weights(message)
    except zmq.Again:
        return None

def receive_weights(node):
    """
    Receive compressed weights from another node and apply them.
    """
    compressed_weights = receive_nonblocking(node.socket)
    if compressed_weights:
        decompressed_weights = decompress_gradients(compressed_weights)
        node.model.set_weights(decompressed_weights)
    else:
        logging.warning(f"No weights received for node {node.node_id}.")
