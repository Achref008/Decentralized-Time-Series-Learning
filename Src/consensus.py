import numpy as np
import logging

class ByzantineFaultTolerance:
    """
    Byzantine Fault Tolerance (BFT) aggregation.
    It ignores extreme faulty model updates using the median.
    """

    def __init__(self, total_nodes, fault_tolerant_nodes):
        self.total_nodes = total_nodes
        self.fault_tolerant_nodes = fault_tolerant_nodes
        self.threshold = self.total_nodes - self.fault_tolerant_nodes

    def aggregate_weights(self, weights_list):
        """
        Combine weights from all nodes using a robust median strategy.
        """
        num_layers = len(weights_list[0])
        aggregated_weights = [np.zeros_like(weights_list[0][i]) for i in range(num_layers)]

        for i in range(num_layers):
            layer_weights = [w[i] for w in weights_list]
            valid_weights = self._filter_faulty_weights(layer_weights)

            if len(valid_weights) >= self.threshold:
                aggregated_weights[i] = np.median(valid_weights, axis=0)
            else:
                logging.warning(
                    f"Not enough reliable weights for layer {i}. Using median of all weights."
                )
                aggregated_weights[i] = np.median(layer_weights, axis=0)

        logging.info("BFT aggregation completed.")
        return aggregated_weights

    def _filter_faulty_weights(self, weights):
        """
        Remove extreme outlier weights that are far from the median.
        """
        median_weight = np.median(weights, axis=0)
        valid_weights = []

        for w in weights:
            max_distance = np.max(np.abs(w - median_weight))
            if max_distance < 0.1:
                valid_weights.append(w)

        logging.info(f"Accepted {len(valid_weights)} / {len(weights)} weights as valid.")
        return valid_weights
