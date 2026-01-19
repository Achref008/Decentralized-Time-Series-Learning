# Robust Decentralized Time-Series Learning  

This repository implements a **decentralized federated learning (DFL)** approach for **collaborative time-series forecasting** on distributed nodes.
Each node trains a local TensorFlow model to predict blower energy consumption and periodically **exchanges model weights** with neighboring nodes in a **peer-to-peer (P2P) network using ZeroMQ**.  
There is **no central server** — all learning is fully decentralized.
To improve reliability in distributed environments, the system uses a **median-based Byzantine Fault Tolerant (BFT) aggregation rule**, making it resilient to faulty, noisy, or misbehaving nodes.

---

## Why this project?

Real-world distributed AI systems often face:

- **Local/private data** (raw data cannot be shared)
- **Unreliable or constrained networks**
- **Heterogeneous or unstable nodes**
- **Risk of faulty or misbehaving participants**

This project demonstrates how decentralized, peer-to-peer learning can:

- Train collaboratively **without a parameter server**
- Remain robust using **Byzantine Fault Tolerance (BFT)**
- Continue learning even when some nodes are slow or disconnected
- Share knowledge **without ever sharing raw data**

---

## How it works (one learning loop)

Each node repeatedly executes the following steps:

1. **Load local time-series data**
2. **Train locally** on past consumption windows
3. **Broadcast model weights** to neighbors via ZeroMQ (PUB socket)
4. **Receive weights** from peers (non-blocking SUB sockets)
5. **Aggregate received models** using a **median-based Byzantine-robust rule**
6. **Update the local model** with the aggregated weights
7. Continue training with improved parameters

**Result:**  
Each node predicts energy consumption locally while **learning collectively** with the rest of the network.

---

## Key features

- **Fully decentralized (serverless) training**
- **Peer-to-peer communication with ZeroMQ (PUB/SUB)**
- **Byzantine Fault Tolerance (BFT)** via median aggregation
- **Non-blocking communication** (nodes don’t freeze if a peer is slow)
- **Time-series forecasting using sliding windows**
- **TensorFlow/Keras regression model**
- **Reproducible experiments** via fixed random seeds
- **Built-in visualization** of data distribution and validation loss

---

## Dataset

The project is designed to work directly with:
data/blower_energy_consumption.csv

This dataset must contain at least these columns:
date | time | consumption

---

Your code will:
- Merge `date` + `time` into a single timestamp  
- Use `consumption` as the prediction target  
- Create sliding windows of length `LOOK_BACK = 24`

---


## Repository structure

```text
EdgeDFL/
│
├── data/
│   └── blower_energy_consumption.csv
│
├── src/
│   ├── main.py               # Entry point
│   ├── config.py             # Hyperparameters & network topology
│   ├── data_loader.py        # CSV loading & cleaning
│   ├── preprocessing.py      # Sliding windows & scaling
│   ├── model.py              # TensorFlow neural network
│   ├── node.py               # Decentralized node logic
│   ├── communication.py      # ZeroMQ messaging
│   ├── consensus.py          # Byzantine aggregation
│   └── visualization.py      # Plots & diagnostics
│
└── README.md
```

---

## Getting started

### 1) Install dependencies


```bash
pip install -r requirements.txt
(If you don’t have one yet, your requirements should include at least:)

- nginx
- Copy code
- tensorflow
- numpy
- pandas
- scikit-learn
- pyzmq
- matplotlib
- tqdm

---

## Getting started

### 1) Install dependencies


```bash
pip install -r requirements.txt
(If you don’t have one yet, your requirements should include at least:)

- nginx
- Copy code
- tensorflow
- numpy
- pandas
- scikit-learn
- pyzmq
- matplotlib
- tqdm
```

---

### 2) Add your dataset

Place your CSV file in:

data/blower_energy_consumption.csv
(You can also edit DATA_PATH in src/config.py if needed.)

---

### 3) Run (local multi-node simulation)
python src/main.py

This will:

- Spawn multiple nodes as threads
- Train them in parallel
- Exchange weights via ZeroMQ
- Apply Byzantine-robust aggregation
- Plot validation loss per node

Note:
In this version, nodes run as threads in one process (simulation).
To run on multiple machines, launch main.py separately per node and configure real IPs/ports.

---

Limitations & notes
- Deployment: Currently runs as a threaded local simulation; multi-process or multi-machine deployment would require additional orchestration.
- Aggregation: Uses median-based BFT, which is robust to outliers but not a full quorum/blockchain consensus protocol.
