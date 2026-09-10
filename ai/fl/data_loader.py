import pandas as pd
import numpy as np


FEATURE_COLUMNS = [

    "speedKmh",
    "accelerationX",
    "accelerationY",
    "accelerationZ",
    "gyroX",
    "gyroY",
    "gyroZ",
    "timeSincePreviousReadingSec",
    "speedChangeKmh",
    "speedAccelerationMs2",
    "gyroMagnitudeRadS"
]


def load_rider_data(csv_path):

    df = pd.read_csv(csv_path)

    # Create binary risk label
    df["riskLabel"] = (
        df["eventType"].fillna("NONE") != "NONE"
    ).astype(int)

    # Keep required features
    X = df[FEATURE_COLUMNS].copy()

    # Replace missing sensor values
    X = X.fillna(0)

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        0
    )

    y = df["riskLabel"].values

    return X.values, y