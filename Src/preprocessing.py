import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import LOOK_BACK, TEST_SIZE, RANDOM_STATE


def feature_engineering(df):
    """
    Turn a single time-series column into supervised samples:

    X[i] = last LOOK_BACK values
    y[i] = next value (the target)

    This is a simple baseline for time-series forecasting.
    """
    if "consumption" not in df.columns:
        raise KeyError("Expected a 'consumption' column in the dataset.")

    X, y = [], []
    values = df["consumption"].values

    for i in range(LOOK_BACK, len(values)):
        X.append(values[i - LOOK_BACK : i])
        y.append(values[i])

    X = np.array(X)
    y = np.array(y)

    logging.info("Feature engineering completed.")
    return X, y


def split_data(X, y):
    """
    Split into train/test sets so we can track performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logging.info("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    """
    Normalize input features using StandardScaler.

    Important: fit scaler on train only, then apply to test.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    logging.info("Data scaling completed.")
    return X_train, X_test, scaler


def convert_dtype(X_train, X_test, y_train, y_test):
    """
    TensorFlow runs more efficiently with float32,
    so we convert everything here.
    """
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    logging.info("Converted arrays to float32.")
    return X_train, X_test, y_train, y_test
