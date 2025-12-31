# import pandas as pd
# import numpy as np

# def extract_features_from_buffer(data: list) -> dict:
#     """
#     Nhận list các mẫu [ax, ay, az, current, voltage, temp]
#     Trích xuất đặc trưng thống kê
#     """
#     if len(data) == 0:
#         return {}

#     df = pd.DataFrame(data, columns=["ax", "ay", "az", "current", "voltage", "temp"])

#     def calc_features(arr):
#         arr = np.array(arr, dtype=float)
#         return {
#             "mean": float(arr.mean()),
#             "rms": float(np.sqrt(np.mean(arr**2))),
#             "std": float(arr.std()),
#             "peak": float(np.max(np.abs(arr)))
#         }

#     features = {}
#     for col in ["ax", "ay", "az", "current", "voltage"]:
#         f = calc_features(df[col])
#         features[f"Mean_{col}"] = f["mean"]
#         features[f"RMS_{col}"] = f["rms"]
#         features[f"STD_{col}"] = f["std"]
#         features[f"Peak_{col}"] = f["peak"]

#     features["Mean_temp"] = float(df["temp"].mean())

#     return features
import pandas as pd
import numpy as np

FS = 2000  # tần số lấy mẫu (Hz) → chỉnh đúng theo hệ của bạn

def extract_features_from_buffer(data: list) -> dict:
    if len(data) == 0:
        return {}

    df = pd.DataFrame(data, columns=["ax", "ay", "az", "current", "voltage", "temp"])

    def time_features(arr):
        arr = np.array(arr, dtype=float)
        return {
            "mean": arr.mean(),
            "rms": np.sqrt(np.mean(arr**2)),
            "std": arr.std(),
            "peak": np.max(np.abs(arr))
        }

    def freq_features(arr):
        arr = np.array(arr, dtype=float)
        fft = np.fft.rfft(arr)
        mag = np.abs(fft)
        freqs = np.fft.rfftfreq(len(arr), d=1/FS)

        return {
            "peak_freq": freqs[np.argmax(mag)],
            "spec_energy": np.sum(mag**2)
        }

    features = {}

    for col in ["ax", "ay", "az", "current", "voltage"]:
        tf = time_features(df[col])
        ff = freq_features(df[col])

        features[f"Mean_{col}"] = float(tf["mean"])
        features[f"RMS_{col}"] = float(tf["rms"])
        features[f"STD_{col}"] = float(tf["std"])
        features[f"Peak_{col}"] = float(tf["peak"])

        features[f"PeakFreq_{col}"] = float(ff["peak_freq"])
        features[f"SpecEnergy_{col}"] = float(ff["spec_energy"])

    features["Mean_temp"] = float(df["temp"].mean())

    return features
