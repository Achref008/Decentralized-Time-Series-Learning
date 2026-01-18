import matplotlib.pyplot as plt
import logging


def check_data_distribution(nodes):
    """
    Quick sanity check:
    Plot how training target values are distributed across nodes.
    If one node has very different data, training/aggregation can behave differently.
    """
    plt.figure(figsize=(10, 6))

    for node in nodes:
        values = []
        for _, y in node.train_dataset:
            values.extend(y.numpy())
        plt.hist(values, bins=20, alpha=0.5, label=f"Node {node.node_id}")

    plt.title("Training Data Distribution per Node")
    plt.xlabel("Consumption (scaled)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    logging.info("Data distribution visualized.")


def visualize_loss(nodes):
    """
    Plot validation loss for each node.
    This helps you see whether nodes learn similarly or diverge.
    """
    plt.figure(figsize=(12, 6))

    for node in nodes:
        if node.history is None:
            logging.warning(f"Node {node.node_id} has no training history to plot.")
            continue

        plt.plot(
            node.history.history.get("val_loss", []),
            label=f"Node {node.node_id} val_loss",
            linestyle="dashed",
        )

    plt.title("Validation Loss per Node")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    logging.info("Loss curves plotted.")
