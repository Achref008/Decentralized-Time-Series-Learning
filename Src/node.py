import numpy as np
import logging
import zmq
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model import create_model
from communication import (
    setup_publisher,
    setup_subscribers,
    serialize_weights,
    deserialize_weights,
)
from consensus import ByzantineFaultTolerance
from config import NODE_PORTS, BATCH_SIZE, EPOCHS, RANDOM_STATE


class Node:
    """
    A "node" simulates one participant in decentralized training.

    Each node:
    - has its own local train/test split (here we reuse the same split for simplicity)
    - trains locally
    - exchanges weights with neighbors via ZeroMQ
    - aggregates weights in a robust way (median-based)
    """

    def __init__(self, node_id, context, X_train, y_train, X_test, y_test, initial_weights=None):
        # Make results reproducible
        np.random.seed(RANDOM_STATE)
        tf.random.set_seed(RANDOM_STATE)

        self.node_id = node_id
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        # TensorFlow datasets are convenient for batching + shuffling
        self.train_dataset = self.create_tf_dataset(X_train, y_train)
        self.test_dataset = self.create_tf_dataset(X_test, y_test)

        # Build model and optionally load common initial weights
        self.model = create_model(X_train.shape[1])
        if initial_weights is not None:
            self.model.set_weights(initial_weights)

        # Network / ZMQ configuration
        self.context = context
        self.neighbors = NODE_PORTS[node_id]["subscribe"]
        self.publish_port = NODE_PORTS[node_id]["publish"]

        self.publisher_socket = setup_publisher(context, self.publish_port)
        self.subscriber_sockets = setup_subscribers(context, self.neighbors)

        self.history = None
        logging.info(f"Node {self.node_id} initialized (PUB {self.publish_port}).")

    def create_tf_dataset(self, X, y):
        """
        Create a shuffled + batched dataset for training.
        """
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        ds = ds.shuffle(buffer_size=len(X)).batch(BATCH_SIZE)
        return ds

    def train(self):
        """
        Train locally on this node's data.

        We use:
        - EarlyStopping to avoid wasting epochs once it stops improving
        - ReduceLROnPlateau to make optimization smoother
        """
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

        self.history = self.model.fit(
            self.train_dataset,
            epochs=EPOCHS,
            verbose=0,
            validation_data=self.test_dataset,
            callbacks=[early_stopping, lr_scheduler],
        )

        logging.info(f"Node {self.node_id} completed training.")
        return self.history

    def evaluate(self):
        """
        Evaluate on the test dataset for quick diagnostics.
        """
        loss, mae = self.model.evaluate(self.test_dataset, verbose=0)
        logging.info(f"Node {self.node_id} evaluation: loss={loss:.4f}, mae={mae:.4f}")
        return loss, mae

    def get_weights(self):
        """
        Return current model weights as a list of numpy arrays.
        """
        return self.model.get_weights()

    def set_weights(self, weights):
        """
        Replace local weights with new weights.
        """
        self.model.set_weights(weights)
        logging.info(f"Node {self.node_id} weights updated.")

    def broadcast_weights(self):
        """
        Send this node's weights to neighbors.
        """
        weights = self.get_weights()
        self.publisher_socket.send(serialize_weights(weights))
        logging.info(f"Node {self.node_id} broadcasted weights.")

    def receive_weights(self):
        """
        Receive weights from neighbors (non-blocking).
        If a neighbor doesn't respond, we continue with what we have.
        """
        local_weights = self.get_weights()
        weights_list = [local_weights]

        for sock in self.subscriber_sockets:
            try:
                msg = sock.recv(flags=zmq.NOBLOCK)
                if msg:
                    weights_list.append(deserialize_weights(msg))
                    logging.info(f"Node {self.node_id} received weights from a neighbor.")
            except zmq.Again:
                logging.warning(f"Node {self.node_id} did not receive weights from one neighbor.")

        # Robust aggregation to reduce impact of odd/outlier updates
        bft = ByzantineFaultTolerance(total_nodes=len(weights_list), fault_tolerant_nodes=2)
        try:
            aggregated = bft.aggregate_weights(weights_list)
        except Exception as e:
            logging.error(f"Aggregation failed: {e}. Keeping local weights.")
            aggregated = local_weights

        self.set_weights(aggregated)
