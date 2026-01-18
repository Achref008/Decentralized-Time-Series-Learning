import os

# -------------------------
# DATA PATH
# -------------------------
DATA_PATH = os.path.join("data", "blower_energy_consumption.csv")

# -------------------------
# MODEL HYPERPARAMETERS
# -------------------------
LOOK_BACK = 24
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.0001
L2_REGULARIZATION = 0.01
DROPOUT_RATE = 0.5

# -------------------------
# ZMQ PORTS (NETWORK SETUP)
# -------------------------
NODE_PORTS = {
    0: {"subscribe": [5555, 5556], "publish": 5557},
    1: {"subscribe": [5557, 5556], "publish": 5558},
    2: {"subscribe": [5558, 5555], "publish": 5559},
    3: {"subscribe": [5559, 5555], "publish": 5560},
    4: {"subscribe": [5560, 5556], "publish": 5561},
}

# -------------------------
# LOGGING
# -------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join("logs", "app.log")

# -------------------------
# BYZANTINE FAULT TOLERANCE
# -------------------------
TOTAL_NODES = 5
FAULT_TOLERANT_NODES = 1

# Ensure folders exist
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

if not os.path.exists(DATA_PATH):
    logging.warning(
        f"Dataset not found at {DATA_PATH}. Please place your CSV file in the data/ folder."
    )
