import tensorflow as tf
import logging
import threading
from tqdm import tqdm
import numpy as np

# Enable memory growth to prevent TensorFlow from allocating all GPU memory upfront
def enable_gpu_memory_growth():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as e:
            logging.error(f"Error enabling memory growth for GPU: {e}")

# Ensure GPU usage
def set_device_for_training(node):
    enable_gpu_memory_growth()  # Ensure dynamic memory allocation
    if tf.config.list_physical_devices('GPU'):
        with tf.device('/GPU:0'):
            node.train()
    else:
        node.train()

# Function to compress gradients (before broadcasting)
def compress_gradients(gradients):
    """
    Compress gradients by quantizing them to int8 (for example).
    You can modify this for more advanced compression techniques like pruning or encoding.
    """
    # Quantize gradients to int8
    compressed_grads = [np.int8(grad * 127) for grad in gradients]
    return compressed_grads

# Function to handle training and communication in parallel
def node_operations(node, rounds=10):
    for _ in tqdm(range(rounds), desc=f"Node {node.node_id} Decentralized Rounds"):
        train_thread = threading.Thread(target=set_device_for_training, args=(node,))
        train_thread.start()
        
        # Broadcast compressed weights
        compressed_weights = compress_gradients(node.model.get_weights())  # Compress the gradients before broadcasting
        serialized_weights = serialize_weights(compressed_weights)  # Serialize compressed weights
        node.broadcast_weights(serialized_weights)  # Broadcast compressed weights
        
        train_thread.join()  # Wait for the training to finish before continuing
        node.receive_weights()  # Receive updated weights

        # Ensure the model is still being trained after each round
        if node.model is None or not hasattr(node.model, 'trainable_variables'):
            logging.warning(f"Node {node.node_id} model is not trainable. Skipping training step.")
            break  # Exit loop if the model is not trainable anymore

    logging.info(f"Node {node.node_id} completed all training rounds.")

# Ensure training does not stop after a fixed number of epochs
def train_model(node, epochs=50):
    try:
        # Ensure model is not stopping prematurely
        for epoch in range(epochs):
            logging.info(f"Node {node.node_id} starting epoch {epoch + 1} of {epochs}")
            node.model.fit(node.train_data, node.train_labels, epochs=1, verbose=0)
            
            if epoch % 10 == 0:  # Log progress every 10 epochs
                logging.info(f"Node {node.node_id} epoch {epoch + 1}/{epochs} training completed.")
            
            # Check for early stopping or convergence here (if needed)
            if check_for_convergence(node):
                logging.info(f"Node {node.node_id} has converged. Stopping training.")
                break  # Stop training if the model has converged

    except Exception as e:
        logging.error(f"Error during training: {e}")

def check_for_convergence(node):
    """
    Checks if the model has converged (e.g., validation loss stops improving).
    Modify this function as needed based on your convergence criteria.
    """
    # Placeholder for convergence check, such as validation loss.
    # Return True if convergence is detected, False otherwise.
    return False
