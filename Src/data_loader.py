import pandas as pd
import logging
from config import DATA_PATH
from sklearn.preprocessing import StandardScaler


def load_data():
    """
    Load the CSV dataset from the path defined in config.py.

    Tip: keep your dataset inside a /data folder so the project stays portable.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        logging.info("CSV file loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file: {e}")
        raise


def combine_datetime(df):
    """
    Many datasets store date and time separately.
    This helper merges them into a single datetime column (if both exist).
    """
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], dayfirst=True)
        df = df.drop(columns=["date", "time"])
        logging.info("Combined 'date' and 'time' into 'datetime'.")
    return df


def set_index(df):
    """
    Use datetime as index to make time-series operations easier (plotting, slicing, etc.).
    """
    if "datetime" not in df.columns and df.index.name != "datetime":
        logging.warning("No 'datetime' column found. Index not changed.")
        return df

    if "datetime" in df.columns:
        df = df.set_index("datetime")
    return df


def handle_missing_values(df):
    """
    Ensure there are no missing consumption values.
    For a real project you might forward-fill or interpolate, but 0 is a simple baseline.
    """
    if "consumption" not in df.columns:
        raise KeyError("Expected a 'consumption' column in the dataset.")

    if df["consumption"].isnull().any():
        logging.warning("Missing values found in 'consumption'. Filling with 0.")
        df["consumption"] = df["consumption"].fillna(0)

    return df


def preprocess_data(df):
    """
    Normalize the consumption values.
    StandardScaler makes training more stable by keeping values around mean=0, std=1.
    """
    if "consumption" not in df.columns:
        raise KeyError("Expected a 'consumption' column in the dataset.")

    scaler = StandardScaler()
    df["consumption"] = scaler.fit_transform(df[["consumption"]])
    logging.info("'consumption' column scaled using StandardScaler.")
    return df
