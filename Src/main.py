import logging
import os
import zmq
import threading

from consensus import ByzantineFaultTolerance
from config import LOG_LEVEL, LOG_FORMAT, LOG_FILE, NODE_PORTS
from data_loader import load_data, combine_datetime, set_index, handle_missing_values
from preprocessing import feature_engineering, split_data, preprocess_data, convert_dtype
from node import Node
from visualization import check_data_distribution, visualize_loss


def setup_logging():
    """
    Log to both console and a file in /logs.
    This makes debugging easier when running multiple nodes/threads.
    """
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )
    logging.info("Logging initialized.")


def initialize_nodes(context, X_train, y_train, X_test, y_test):
    """
    Create all nodes and ensure they start from the same initial weights.

    Why:
    - If nodes start from random weights, they drift too far apart quickly.
    - Shared initialization keeps the training comparable across nodes.
    """
    initial_node = Node(0, context, X_train, y_train, X_test, y_test)
    initial_weights = initial_node.get_weights()

    other_nodes = [
        Node(i, context, X_train, y_train, X_test, y_test, initial_weights=initial_weights)
        for i in range(1, len(NODE_PORTS))
    ]
    return initial_node, other_nodes


def node_operations_with_bft(node, nodes):
    """
    One worker routine (run per node in a thread):
    1) Train locally
    2) Collect weights from everyone
    3) Aggregate using a robust (median-based) rule
    4) Update the node with the aggregated result
    """
    logging.info(f"Node {node.node_id} starting training.")
    node.train()

    # Collect weights from all nodes (including itself)
    weights_list = [n.get_weights() for n in nodes]

    # Robust aggregation (helps reduce impact of weird/outlier updates)
    bft = ByzantineFaultTolerance(total_nodes=len(weights_list), fault_tolerant_nodes=1)
    aggregated_weights = bft.aggregate_weights(weights_list)

    node.set_weights(aggregated_weights)
    logging.info(f"Node {node.node_id} finished training + aggregation.")


def main():
    """
    Entry point:
    - Load + prepare data
    - Create nodes
    - Train each node in parallel threads
    - Plot basic results
    """
    setup_logging()

    # ---- Data pipeline ----
    df = load_data()
    df = combine_datetime(df)
    df = set_index(df)
    df = handle_missing_values(df)

    X, y = feature_engineering(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test, _ = preprocess_data(X_train, X_test)
    X_train, X_test, y_train, y_test = convert_dtype(X_train, X_test, y_train, y_test)

    # ---- Communication context ----
    context = zmq.Context()

    # ---- Nodes ----
    initial_node, other_nodes = initialize_nodes(context, X_train, y_train, X_test, y_test)
    all_nodes = [initial_node] + other_nodes

    # ---- Run nodes in parallel ----
    threads = []
    for node in all_nodes:
        t = threading.Thread(target=node_operations_with_bft, args=(node, all_nodes))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    logging.info("All nodes finished.")

    # ---- Simple plots ----
    check_data_distribution(all_nodes)
    visualize_loss(all_nodes)


if __name__ == "__main__":
    main()
