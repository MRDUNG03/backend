import pandas as pd
import numpy as np

def extract_features_from_buffer(data: list) -> dict:
    """
    Nhận list các mẫu [ax, ay, az, current, voltage, temp]
    Trích xuất đặc trưng thống kê
    """
    if len(data) == 0:
        return {}

    df = pd.DataFrame(data, columns=["ax", "ay", "az", "current", "voltage", "temp"])

    def calc_features(arr):
        arr = np.array(arr, dtype=float)
        return {
            "mean": float(arr.mean()),
            "rms": float(np.sqrt(np.mean(arr**2))),
            "std": float(arr.std()),
            "peak": float(np.max(np.abs(arr)))
        }

    features = {}
    for col in ["ax", "ay", "az", "current", "voltage"]:
        f = calc_features(df[col])
        features[f"Mean_{col}"] = f["mean"]
        features[f"RMS_{col}"] = f["rms"]
        features[f"STD_{col}"] = f["std"]
        features[f"Peak_{col}"] = f["peak"]

    features["Mean_temp"] = float(df["temp"].mean())

    return features