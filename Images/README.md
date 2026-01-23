# Experimental Results

The following experiments evaluate the behavior of the fully decentralized multi-node learning system under realistic conditions, including non-IID data splits and independent local training. All results are obtained from 5 cooperative nodes (Node 0–4) communicating through peer-to-peer aggregation without any central server.
## 1. Training & Validation Loss Across Nodes (200 communication rounds)

![Training and Validation Loss](https://github.com/Achref008/DTSL-Decentralized-Time-Series-Learning/blob/main/Images/Training%20Loss.png)
*This figure tracks the learning progress of 5 independent nodes (Node 0 to Node 4) over 200 decentralized synchronization rounds.*

This figure shows the training (dashed) and validation (solid) loss for each node over **200 decentralized communication rounds**.
In this simulation, each node runs as an **independent thread** with its own local dataset and model.
One round consists of **local training followed by peer-to-peer weight exchange and aggregation** (i.e., one epoch = one communication round).

**Observations**

* **Global convergence**: All nodes show a steady decrease in loss, confirming stable optimization and successful decentralized synchronization.
* **Independent learning dynamics**: Nodes follow slightly different trajectories due to heterogeneous local data and stochastic updates.
* **Consensus behavior**: After repeated communication rounds, models gradually align and converge to similar validation losses.
* **Final performance**: By round 200, all nodes stabilize around **MSE ≈ 0.02–0.04**, indicating accurate energy demand prediction.


**Interpretation**
These results demonstrate that repeated **local node trainining** → **communicate** → **aggregate** cycles enable fully decentralized nodes to collaboratively learn a shared model, while remaining robust to statistical differences and without relying on a central coordinator.

---

## 2. Final Validation MSE Comparison

![Energy Distribution](https://github.com/Achref008/DTSL-Decentralized-Time-Series-Learning/blob/main/Images/Energy%20distribution%20across%20nodes.png)


This histogram compares the original energy consumption distribution with the local data observed by each node after preprocessing and dataset partitioning.

Observations

* **Heterogeneous local datasets**:Each node exhibits a distinct consumption profile, with visible shifts in density and value ranges.
* **Non-IID setting**:The distributions are not identical across nodes, reflecting real-world edge scenarios where sensors collect different patterns over time.
* **Preprocessing consistency**: Normalization preserves overall structure while stabilizing scale for training.

**Interpretation**
This confirms that the system operates under **non-IID conditions**, making the learning problem more challenging and realistic.
Successful convergence in the previous plot demonstrates that the decentralized aggregation mechanism effectively handles statistical heterogeneity.

---

**Experimental Configuration**
- **Nodes**: 5 simulated decentralized peers (each executed as an independent thread)
- **Architecture**: Feed-forward neural network (TensorFlow/Keras)
- **Optimizer**: Adam
- **Loss**: Mean Squared Error (MSE)
- **Training scheme**: Local training + periodic synchronization
- **Communication**: ZMQ PUB/SUB peer-to-peer message passing
- **Aggregation**: Robust median-based consensus (Byzantine-tolerant)
- **Setting**: Fully decentralized logic with no central coordinator, simulated on a single machine

**Big picture**
- **The distribution plot highlights the challenge**: each node trains on statistically different (non-IID) local data partitions.
- **The loss plot demonstrates the outcome**: repeated communication rounds enable stable collaborative convergence across all peers.

Together, these results show that even in a **multi-threaded decentralized simulation**, independent nodes can reliably learn a shared model through peer-to-peer aggregation while maintaining consistent performance under heterogeneous data.


