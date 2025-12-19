# features_backend.py
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# ---------- HÀM TÍNH FEATURE 1D CHO MỘT KÊNH ----------
def feat_1d(arr: np.ndarray) -> dict:
    if arr is None or len(arr) == 0:
        return {
            "mean": np.nan, "rms": np.nan, "std": np.nan, "peak": np.nan,
            "kurtosis": np.nan, "skewness": np.nan,
            "crest_factor": np.nan, "shape_factor": np.nan, "impulse_factor": np.nan,
        }

    arr = np.asarray(arr, dtype=float)
    mean = arr.mean()
    rms  = np.sqrt(np.mean(arr**2))
    std  = arr.std()
    peak = np.max(np.abs(arr))

    kurt_val = kurtosis(arr)      # nhấn mạnh xung
    skew_val = skew(arr)          # bất đối xứng
    crest = peak / rms if rms != 0 else np.nan
    mean_abs = np.mean(np.abs(arr))
    shape = rms / mean_abs if mean_abs != 0 else np.nan
    impulse = peak / mean_abs if mean_abs != 0 else np.nan

    return {
        "mean": float(mean),
        "rms": float(rms),
        "std": float(std),
        "peak": float(peak),
        "kurtosis": float(kurt_val),
        "skewness": float(skew_val),
        "crest_factor": float(crest),
        "shape_factor": float(shape),
        "impulse_factor": float(impulse),
    }

# ---------- HÀM TRÍCH ĐẶC TRƯNG CHO 1 SEGMENT (1 CỬA SỔ) ----------
def extract_features_from_segment(seg_df: pd.DataFrame) -> dict:
    # Lấy các kênh, nếu thiếu thì cho mảng rỗng
    ax = seg_df["ax"].values       if "ax" in seg_df.columns else np.array([])
    ay = seg_df["ay"].values       if "ay" in seg_df.columns else np.array([])
    az = seg_df["az"].values       if "az" in seg_df.columns else np.array([])
    cur = seg_df["current"].values if "current" in seg_df.columns else np.array([])
    vol = seg_df["voltage"].values if "voltage" in seg_df.columns else np.array([])
    tmp = seg_df["temp"].values    if "temp" in seg_df.columns else np.array([])

    f_ax  = feat_1d(ax)
    f_ay  = feat_1d(ay)
    f_az  = feat_1d(az)
    f_cur = feat_1d(cur)
    f_vol = feat_1d(vol)

    # Temp: dùng lại feat_1d + thêm min/max/range
    if len(tmp) > 0:
        tmp = np.asarray(tmp, dtype=float)
        f_tmp = feat_1d(tmp)
        min_temp = float(np.min(tmp))
        max_temp = f_tmp["peak"]     # peak = max(|tmp|)
        # Nếu bạn muốn max_temp = max(tmp) (không abs) thì đổi lại:
        # max_temp = float(np.max(tmp))
        range_temp = max_temp - min_temp
    else:
        f_tmp = feat_1d(tmp)
        min_temp = np.nan
        max_temp = np.nan
        range_temp = np.nan

    # ========== ĐỊNH NGHĨA BỘ FEATURE CHUẨN ==========

    features = {
        # --- ax ---
        "Mean_ax": f_ax["mean"], "RMS_ax": f_ax["rms"], "STD_ax": f_ax["std"], "Peak_ax": f_ax["peak"],
        "Kurt_ax": f_ax["kurtosis"], "Skew_ax": f_ax["skewness"],
        "Crest_ax": f_ax["crest_factor"], "Shape_ax": f_ax["shape_factor"], "Impulse_ax": f_ax["impulse_factor"],

        # --- ay ---
        "Mean_ay": f_ay["mean"], "RMS_ay": f_ay["rms"], "STD_ay": f_ay["std"], "Peak_ay": f_ay["peak"],
        "Kurt_ay": f_ay["kurtosis"], "Skew_ay": f_ay["skewness"],
        "Crest_ay": f_ay["crest_factor"], "Shape_ay": f_ay["shape_factor"], "Impulse_ay": f_ay["impulse_factor"],

        # --- az ---
        "Mean_az": f_az["mean"], "RMS_az": f_az["rms"], "STD_az": f_az["std"], "Peak_az": f_az["peak"],
        "Kurt_az": f_az["kurtosis"], "Skew_az": f_az["skewness"],
        "Crest_az": f_az["crest_factor"], "Shape_az": f_az["shape_factor"], "Impulse_az": f_az["impulse_factor"],

        # --- current ---
        "Mean_current": f_cur["mean"], "RMS_current": f_cur["rms"], "STD_current": f_cur["std"], "Peak_current": f_cur["peak"],
        "Kurt_current": f_cur["kurtosis"], "Skew_current": f_cur["skewness"],
        "Crest_current": f_cur["crest_factor"], "Shape_current": f_cur["shape_factor"], "Impulse_current": f_cur["impulse_factor"],

        # --- voltage ---
        "Mean_voltage": f_vol["mean"], "RMS_voltage": f_vol["rms"], "STD_voltage": f_vol["std"], "Peak_voltage": f_vol["peak"],
        "Kurt_voltage": f_vol["kurtosis"], "Skew_voltage": f_vol["skewness"],
        "Crest_voltage": f_vol["crest_factor"], "Shape_voltage": f_vol["shape_factor"], "Impulse_voltage": f_vol["impulse_factor"],

        # --- temp ---
        "Mean_temp": f_tmp["mean"],
        "STD_temp": f_tmp["std"],
        "Max_temp": max_temp,
        "Min_temp": min_temp,
        "Range_temp": range_temp,
    }

    return features

# List cột feature dùng chung cho train + backend
FEATURE_COLS = list(extract_features_from_segment(pd.DataFrame({
    "ax":[0,1], "ay":[0,1], "az":[0,1],
    "current":[0,1], "voltage":[0,1], "temp":[0,1]
})).keys())
