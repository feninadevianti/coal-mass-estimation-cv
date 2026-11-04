"""
SCRIPT 18 - PERBANDINGAN MODEL ESTIMASI MASSA BATUBARA
========================================================

Script gabungan untuk mencoba beberapa model estimasi massa muatan batubara
pada gerbong/wagon berdasarkan data kalibrasi Script 18.

Input utama:
    hasil_18_kalibrasi_empiris_massa/18_data_kalibrasi_empiris_massa_per_baseline.csv

Jika input belum ada, script membuat template dari:
    hasil_volume_homografi/volume_history.csv

Output utama:
    hasil_18_perbandingan_5_model_estimasi_massa/
        01_density_manual/
        02_density_empirical/
        03_linear_regression/
        04_piecewise_linear/
        05_multiple_regression_profile/
        00_ringkasan_hasil/

Folder ringkasan berisi:
    - perbandingan metrik model
    - ranking metode terbaik berbasis data
    - grafik gabungan model
    - laporan PDF
    - laporan DOCX jika python-docx tersedia
    - daftar data yang perlu dicek ulang karena error tinggi

Catatan:
- Analisis metode terbaik dihitung dari MAE, RMSE, MAPE, R2, dan jumlah error tinggi.
- Data yang disarankan dicek ulang bukan otomatis dibuang. Cek ulang berarti periksa
  kembali timbangannya, mask batubara, volume/depth, baseline, dan bentuk muatannya.
"""

import os
import csv
import json
import math
import shutil
from datetime import datetime
from collections import defaultdict

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

try:
    from docx import Document
    from docx.shared import Inches
    DOCX_OK = True
except Exception:
    DOCX_OK = False


# =========================================================
# KONFIGURASI
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOLUME_HISTORY_CSV = os.path.join(BASE_DIR, "hasil_volume_homografi", "volume_history.csv")
CALIBRATION_DIR = os.path.join(BASE_DIR, "hasil_18_kalibrasi_empiris_massa")
CALIBRATION_CSV = os.path.join(CALIBRATION_DIR, "18_data_kalibrasi_empiris_massa_per_baseline.csv")

OUTPUT_ROOT = os.path.join(BASE_DIR, "hasil_18_perbandingan_5_model_estimasi_massa")
SUMMARY_DIR = os.path.join(OUTPUT_ROOT, "00_ringkasan_hasil")

USE_ZERO_ACTUAL_FOR_FIT = True
TRAIN_RATIO = 0.80
RANDOM_SEED = 42
FALLBACK_AREA_PER_PIXEL_CM2 = 0.01  # top-view Script 12: 10 px/cm -> 0,1 cm x 0,1 cm
MANUAL_DENSITY_G_PER_CM3 = 0.83
MANUAL_DENSITY_BY_BASELINE = {
    # Contoh jika ingin beda baseline:
    # "10": 1.20,
    # "15": 1.18,
}

RIDGE_ALPHA = 1.0
HUBER_DELTA = 1.345
ROBUST_MAX_ITER = 100
ROBUST_TOL = 1e-8

PIECEWISE_SEGMENTS = 3
SPLINE_KNOT_QUANTILES = [0.25, 0.50, 0.75]

# Ambang awal untuk menandai data perlu cek ulang.
# Script juga memakai ambang adaptif dari MAE metode terbaik.
ABS_ERROR_RECHECK_GRAM = 150.0
PERCENT_ERROR_RECHECK = 15.0


METHODS = [
    ("01_density_manual", "Densitas Manual", "density_manual"),
    ("02_density_empirical", "Densitas Empiris", "density_empirical"),
    ("03_linear_regression", "Linear Regression", "linear"),
    ("04_piecewise_linear", "Piecewise Linear", "piecewise"),
    ("05_multiple_regression_profile", "Multiple Regression + Profile", "multiple_profile"),
    ("06_piecewise_linear_profile", "Piecewise Linear + Profile", "piecewise_profile"),
]

ENHANCED_NUMERIC_FEATURES = [
    "volume_cm3",
    "mean_height_cm",
    "median_height_cm",
    "max_height_cm",
    "valid_pixels",
    "area_batubara_cm2",
    "p95_height_cm",
    "p98_height_cm",
]


METHOD_DETAILS = {
    "linear": {
        "rumus": "massa = a*volume_cm3 + b",
        "penjelasan": "Model garis lurus antara volume hasil sistem dan massa aktual.",
        "relevansi": "Dipakai sebagai model dasar karena massa batubara secara fisik berkaitan dengan volume.",
    },
    "poly2": {
        "rumus": "massa = a*volume_cm3^2 + b*volume_cm3 + c",
        "penjelasan": "Model kurva orde 2 yang memberi ruang hubungan volume dan massa tidak selalu lurus.",
        "relevansi": "Relevan jika error berubah pada muatan kecil, sedang, dan besar.",
    },
    "density_empirical": {
        "rumus": "massa = densitas_empiris*volume_cm3",
        "penjelasan": "Densitas dihitung dari data aktual, lalu dipakai untuk mengubah volume menjadi massa.",
        "relevansi": "Sesuai konsep fisika massa = densitas x volume, tetapi densitas diambil dari data kalibrasi.",
    },
    "density_manual": {
        "rumus": "massa = densitas_manual*volume_cm3",
        "penjelasan": "Densitas ditentukan manual oleh peneliti, bukan dihitung dari data.",
        "relevansi": "Berguna untuk menguji asumsi densitas batubara tertentu terhadap hasil sistem.",
    },
    "multiple": {
        "rumus": "massa = b0 + b1*volume + b2*mean_height + b3*max_height + b4*valid_pixels",
        "penjelasan": "Model memakai beberapa fitur sistem, bukan hanya volume.",
        "relevansi": "Relevan karena massa dapat dipengaruhi tinggi rata-rata, tinggi maksimum, dan luas area valid.",
    },
    "multiple_profile": {
        "rumus": "massa = multiple regression + variabel profile muatan",
        "penjelasan": "Model menambahkan kategori bentuk muatan seperti rata, gundukan, atau asimetris.",
        "relevansi": "Relevan karena bentuk muatan dapat mengubah distribusi tinggi dan akurasi volume.",
    },
    "piecewise": {
        "rumus": "massa = a_segmen*volume_cm3 + b_segmen",
        "penjelasan": "Model membagi volume menjadi segmen rendah, sedang, dan tinggi, lalu membuat garis berbeda.",
        "relevansi": "Relevan jika hubungan volume dan massa berbeda pada muatan kecil, sedang, dan besar.",
    },
    "piecewise_profile": {
        "rumus": "massa = a_segmen*volume_cm3 + b_segmen + offset_profile",
        "penjelasan": "Model membagi data menjadi segmen volume, lalu menambahkan koreksi berdasarkan profil muatan.",
        "relevansi": "Relevan karena muatan rata, gundukan, dan asimetris dapat memiliki pola volume-massa yang berbeda.",
    },
    "robust": {
        "rumus": "massa = a*volume_cm3 + b dengan bobot Huber",
        "penjelasan": "Model linear yang menurunkan pengaruh data dengan residual sangat besar.",
        "relevansi": "Relevan saat ada data ekstrem yang dapat menarik garis regresi terlalu jauh.",
    },
    "poly_ridge": {
        "rumus": "massa = intercept + b1*z + b2*z^2, dengan Ridge",
        "penjelasan": "Polynomial orde 2 yang distabilkan dengan regularisasi Ridge.",
        "relevansi": "Relevan untuk menangkap pola melengkung tanpa membuat koefisien terlalu liar.",
    },
    "spline_ridge": {
        "rumus": "massa = intercept + b1*z + sum(h_i*max(0,z-k_i))",
        "penjelasan": "Model spline yang membentuk garis lentur berdasarkan knot volume dan distabilkan Ridge.",
        "relevansi": "Relevan jika pola massa-volume berubah halus pada beberapa rentang muatan.",
    },
}


# =========================================================
# UTILITAS FILE DAN PARSING
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float(value, default=None):
    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.lower() in ["nan", "none", "null"]:
        return default
    try:
        return float(text.replace(",", "."))
    except Exception:
        return default


def parse_baseline(value):
    val = parse_float(value, None)
    if val is None:
        return "unknown"
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    return str(val).replace(".", "_")


def baseline_sort_key(value):
    val = parse_float(str(value).replace("_", "."), None)
    return (0, val) if val is not None else (1, str(value))


def get_actual_weight(row):
    for key in ["massa_aktual_gram", "massa_aktual_g", "actual_weight_g"]:
        val = parse_float(row.get(key), None)
        if val is not None:
            return val
    return None


def get_target_mass(row):
    for key in ["target_massa_gram", "target_massa_g", "target_weight_g"]:
        val = parse_float(row.get(key), None)
        if val is not None:
            return val
    return None


def get_volume_cm3(row):
    for key in ["volume_cm3", "volume_est_cm3"]:
        val = parse_float(row.get(key), None)
        if val is not None:
            return val
    return None


def get_first_float(row, keys, default=None):
    for key in keys:
        val = parse_float(row.get(key), None)
        if val is not None:
            return val
    return default


def get_area_batubara_cm2(row, valid_pixels):
    area_direct = get_first_float(row, ["area_batubara_cm2", "coal_area_cm2", "luas_batubara_cm2"], None)
    if area_direct is not None:
        return area_direct
    if valid_pixels is None:
        return None
    area_per_pixel_m2 = get_first_float(row, ["area_per_pixel_m2"], None)
    if area_per_pixel_m2 is not None:
        return float(valid_pixels) * float(area_per_pixel_m2) * 10000.0
    return float(valid_pixels) * FALLBACK_AREA_PER_PIXEL_CM2


def normalize_profile(value):
    text = str(value or "").strip().lower()
    if text == "":
        return "unknown"
    text = text.replace(" ", "_")
    return "".join(ch for ch in text if ch.isalnum() or ch == "_") or "unknown"


def normalize_volume_history_row(row, no):
    return {
        "no": no,
        "timestamp": row.get("timestamp", ""),
        "sample_id": row.get("sample_id", row.get("prefix", f"sample_{no:03d}")),
        "prefix": row.get("prefix", ""),
        "baseline_cm": row.get("baseline_cm", ""),
        "target_massa_gram": row.get("target_massa_gram", row.get("target_weight_g", "")),
        "massa_aktual_gram": row.get("massa_aktual_gram", row.get("actual_weight_g", "")),
        "condition": row.get("condition", ""),
        "profile": row.get("profile", ""),
        "repetition": row.get("repetition", ""),
        "volume_cm3": row.get("volume_cm3", ""),
        "volume_liter": row.get("volume_liter", ""),
        "volume_m3": row.get("volume_m3", ""),
        "mean_height_cm": row.get("mean_height_cm", ""),
        "median_height_cm": row.get("median_height_cm", ""),
        "max_height_cm": row.get("max_height_cm", ""),
        "p95_height_cm": row.get("p95_height_cm", row.get("p95_cm", "")),
        "p98_height_cm": row.get("p98_height_cm", row.get("p98_cm", "")),
        "valid_pixels": row.get("valid_pixels", row.get("valid_volume_pixels", "")),
        "area_per_pixel_m2": row.get("area_per_pixel_m2", ""),
        "area_batubara_cm2": row.get("area_batubara_cm2", ""),
        "status": row.get("status", ""),
        "catatan": "isi massa_aktual_gram sesuai timbangan digital" if get_actual_weight(row) is None else "",
    }


# =========================================================
# LOAD DATA
# =========================================================
def create_template_if_needed():
    ensure_dir(CALIBRATION_DIR)
    if os.path.exists(CALIBRATION_CSV):
        return False
    if not os.path.exists(VOLUME_HISTORY_CSV):
        raise FileNotFoundError(
            "Data kalibrasi belum ada dan volume_history.csv belum ditemukan.\n"
            f"Data kalibrasi dicari: {CALIBRATION_CSV}\n"
            f"Volume history dicari : {VOLUME_HISTORY_CSV}"
        )
    rows = read_csv(VOLUME_HISTORY_CSV)
    if not rows:
        raise RuntimeError("volume_history.csv kosong.")
    out_rows = [normalize_volume_history_row(row, i) for i, row in enumerate(rows, start=1)]
    write_csv(CALIBRATION_CSV, out_rows, list(out_rows[0].keys()))
    print("[OK] Template data kalibrasi dibuat.")
    print(f"Isi kolom massa_aktual_gram di: {CALIBRATION_CSV}")
    return True


def load_calibration_data():
    created = create_template_if_needed()
    if created:
        return []

    rows = read_csv(CALIBRATION_CSV)
    data = []
    for i, row in enumerate(rows, start=1):
        volume = get_volume_cm3(row)
        actual = get_actual_weight(row)
        if volume is None or actual is None:
            continue
        if volume < 0 or actual < 0:
            continue
        if actual == 0 and not USE_ZERO_ACTUAL_FOR_FIT:
            continue
        baseline_key = parse_baseline(row.get("baseline_cm", ""))
        mean_height_cm = parse_float(row.get("mean_height_cm"), None)
        median_height_cm = parse_float(row.get("median_height_cm"), None)
        max_height_cm = parse_float(row.get("max_height_cm"), None)
        valid_pixels = parse_float(row.get("valid_pixels"), None)
        p95_height_cm = get_first_float(row, ["p95_height_cm", "p95_cm"], max_height_cm)
        p98_height_cm = get_first_float(row, ["p98_height_cm", "p98_cm"], max_height_cm)
        area_batubara_cm2 = get_area_batubara_cm2(row, valid_pixels)
        data.append({
            "no": row.get("no", i),
            "timestamp": row.get("timestamp", ""),
            "sample_id": row.get("sample_id", row.get("prefix", f"sample_{i:03d}")),
            "prefix": row.get("prefix", ""),
            "baseline_key": baseline_key,
            "baseline_cm": row.get("baseline_cm", baseline_key),
            "target_weight_g": get_target_mass(row),
            "actual_weight_g": float(actual),
            "condition": row.get("condition", ""),
            "profile": row.get("profile", ""),
            "profile_norm": normalize_profile(row.get("profile", "")),
            "repetition": row.get("repetition", ""),
            "volume_cm3": float(volume),
            "volume_liter": parse_float(row.get("volume_liter"), None),
            "volume_m3": parse_float(row.get("volume_m3"), None),
            "mean_height_cm": mean_height_cm,
            "median_height_cm": median_height_cm,
            "max_height_cm": max_height_cm,
            "p95_height_cm": p95_height_cm,
            "p98_height_cm": p98_height_cm,
            "valid_pixels": valid_pixels,
            "area_batubara_cm2": area_batubara_cm2,
            "status": row.get("status", ""),
            "catatan": row.get("catatan", ""),
        })
    return data


def group_by_baseline(data):
    groups = defaultdict(list)
    for row in data:
        groups[row["baseline_key"]].append(row)
    return dict(groups)


# =========================================================
# METRIK
# =========================================================
def compute_metrics(y, pred):
    y = np.array(y, dtype=float)
    pred = np.array(pred, dtype=float)
    err = pred - y
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(math.sqrt(np.mean(err ** 2)))
    mask = y > 0
    mape = float(np.mean(abs_err[mask] / y[mask]) * 100.0) if np.any(mask) else None
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "mae_gram": mae,
        "rmse_gram": rmse,
        "mape_percent": mape,
        "r2": r2,
        "error": err,
        "abs_error": abs_err,
    }


def count_high_errors(y, pred, mae):
    threshold = max(ABS_ERROR_RECHECK_GRAM, 1.5 * mae)
    y = np.array(y, dtype=float)
    pred = np.array(pred, dtype=float)
    abs_err = np.abs(pred - y)
    percent = np.where(y > 0, abs_err / y * 100.0, 0.0)
    return int(np.sum((abs_err >= threshold) | (percent >= PERCENT_ERROR_RECHECK)))


def aggregate_method_metrics(pred_rows):
    y = np.array([r["actual_weight_g"] for r in pred_rows], dtype=float)
    pred = np.array([r["pred_weight_g"] for r in pred_rows], dtype=float)
    metrics = compute_metrics(y, pred)
    metrics.pop("error", None)
    metrics.pop("abs_error", None)
    metrics["jumlah_data"] = len(pred_rows)
    metrics["high_error_count"] = count_high_errors(y, pred, metrics["mae_gram"])
    return metrics


# =========================================================
# MODEL FITTING
# =========================================================
def fit_linear(rows):
    x = np.array([r["volume_cm3"] for r in rows], dtype=float)
    y = np.array([r["actual_weight_g"] for r in rows], dtype=float)
    a, b = np.polyfit(x, y, 1)
    pred = np.maximum(a * x + b, 0.0)
    return pred, {"formula": "massa = a*volume_cm3 + b", "a": float(a), "b": float(b)}


def fit_poly2(rows):
    x = np.array([r["volume_cm3"] for r in rows], dtype=float)
    y = np.array([r["actual_weight_g"] for r in rows], dtype=float)
    a, b, c = np.polyfit(x, y, 2)
    pred = np.maximum(a * x ** 2 + b * x + c, 0.0)
    return pred, {
        "formula": "massa = a*volume_cm3^2 + b*volume_cm3 + c",
        "a": float(a),
        "b": float(b),
        "c": float(c),
    }


def fit_density_empirical(rows):
    x = np.array([r["volume_cm3"] for r in rows], dtype=float)
    y = np.array([r["actual_weight_g"] for r in rows], dtype=float)
    valid = x > 0
    density = float(np.median(y[valid] / x[valid])) if np.any(valid) else 0.0
    pred = np.maximum(density * x, 0.0)
    return pred, {"formula": "massa = density_g_per_cm3 * volume_cm3", "density_g_per_cm3": density, "density_mode": "median"}


def fit_density_manual(rows, baseline_key):
    x = np.array([r["volume_cm3"] for r in rows], dtype=float)
    density = float(MANUAL_DENSITY_BY_BASELINE.get(str(baseline_key), MANUAL_DENSITY_G_PER_CM3))
    pred = np.maximum(density * x, 0.0)
    return pred, {"formula": "massa = manual_density_g_per_cm3 * volume_cm3", "density_g_per_cm3": density}


def numeric_matrix(rows, features, existing_stats=None):
    cols = []
    stats = {}
    for feature in features:
        vals = np.array([np.nan if r.get(feature) is None else float(r.get(feature)) for r in rows], dtype=float)
        valid = np.isfinite(vals)
        if existing_stats and feature in existing_stats:
            fill = float(existing_stats[feature]["fill"])
            mean = float(existing_stats[feature]["mean"])
            std = float(existing_stats[feature]["std"])
        else:
            fill = float(np.median(vals[valid])) if np.any(valid) else 0.0
            vals_for_stats = np.where(valid, vals, fill)
            mean = float(np.mean(vals_for_stats))
            std = float(np.std(vals_for_stats))
            if std <= 1e-12:
                std = 1.0
        vals = np.where(valid, vals, fill)
        cols.append((vals - mean) / std)
        stats[feature] = {"fill": fill, "mean": mean, "std": std}
    return np.column_stack(cols), stats


def fit_multiple(rows, include_profile=False):
    features = ENHANCED_NUMERIC_FEATURES
    y = np.array([r["actual_weight_g"] for r in rows], dtype=float)
    x_num, stats = numeric_matrix(rows, features)
    feature_names = ["intercept"] + features
    cols = [np.ones(len(rows))]
    cols.extend([x_num[:, i] for i in range(x_num.shape[1])])
    profile_reference = None
    profile_levels = []
    if include_profile:
        profiles = sorted(set(r["profile_norm"] for r in rows))
        if profiles:
            profile_reference = profiles[0]
            profile_levels = profiles[1:]
            for level in profile_levels:
                cols.append(np.array([1.0 if r["profile_norm"] == level else 0.0 for r in rows], dtype=float))
                feature_names.append(f"profile_{level}")
    x = np.column_stack(cols)
    coef, residuals, rank, singular_values = np.linalg.lstsq(x, y, rcond=None)
    pred = np.maximum(x @ coef, 0.0)
    return pred, {
        "formula": "massa = intercept + sum(coef_i * fitur_terstandar_i)",
        "features": feature_names,
        "numeric_features": features,
        "coefficients": {name: float(value) for name, value in zip(feature_names, coef)},
        "numeric_standardization": stats,
        "include_profile": include_profile,
        "profile_reference": profile_reference,
        "profile_levels": profile_levels,
        "rank": int(rank),
        "singular_values": [float(v) for v in singular_values],
    }


def piecewise_breaks(volume):
    if PIECEWISE_SEGMENTS <= 1:
        return []
    qs = [i / PIECEWISE_SEGMENTS for i in range(1, PIECEWISE_SEGMENTS)]
    return sorted(set(float(np.quantile(volume, q)) for q in qs))


def segment_index(value, breaks):
    idx = 0
    for br in breaks:
        if value > br:
            idx += 1
    return idx


def fit_piecewise(rows):
    x = np.array([r["volume_cm3"] for r in rows], dtype=float)
    y = np.array([r["actual_weight_g"] for r in rows], dtype=float)
    breaks = piecewise_breaks(x)
    pred = np.zeros_like(y)
    segments = {}
    fallback_a, fallback_b = np.polyfit(x, y, 1)
    for seg in range(len(breaks) + 1):
        mask = np.array([segment_index(v, breaks) == seg for v in x], dtype=bool)
        if np.sum(mask) >= 2:
            a, b = np.polyfit(x[mask], y[mask], 1)
        else:
            a, b = fallback_a, fallback_b
        pred[mask] = a * x[mask] + b
        segments[str(seg + 1)] = {
            "count": int(np.sum(mask)),
            "a": float(a),
            "b": float(b),
            "volume_min": float(np.min(x[mask])) if np.any(mask) else None,
            "volume_max": float(np.max(x[mask])) if np.any(mask) else None,
            "formula": "massa = a*volume_cm3 + b",
        }
    pred = np.maximum(pred, 0.0)
    return pred, {"formula": "massa = a_segment*volume_cm3 + b_segment", "breaks_volume_cm3": breaks, "segments": segments}


def fit_piecewise_profile(rows):
    x = np.array([r["volume_cm3"] for r in rows], dtype=float)
    y = np.array([r["actual_weight_g"] for r in rows], dtype=float)
    breaks = piecewise_breaks(x)
    pred = np.zeros_like(y)
    profiles = sorted(set(r["profile_norm"] for r in rows))
    profile_reference = profiles[0] if profiles else None
    profile_levels = profiles[1:] if profiles else []
    segments = {}
    fallback_a, fallback_b = np.polyfit(x, y, 1)

    for seg in range(len(breaks) + 1):
        mask = np.array([segment_index(v, breaks) == seg for v in x], dtype=bool)
        n = int(np.sum(mask))
        if n >= max(2, 2 + len(profile_levels)):
            cols = [np.ones(n), x[mask]]
            for level in profile_levels:
                cols.append(np.array([1.0 if r["profile_norm"] == level else 0.0 for r, use in zip(rows, mask) if use], dtype=float))
            design = np.column_stack(cols)
            coef, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
        elif n >= 2:
            a, b = np.polyfit(x[mask], y[mask], 1)
            coef = np.array([b, a] + [0.0 for _ in profile_levels], dtype=float)
        else:
            coef = np.array([fallback_b, fallback_a] + [0.0 for _ in profile_levels], dtype=float)

        cols_all = [np.ones(n), x[mask]]
        for level in profile_levels:
            cols_all.append(np.array([1.0 if r["profile_norm"] == level else 0.0 for r, use in zip(rows, mask) if use], dtype=float))
        design_all = np.column_stack(cols_all) if n > 0 else np.empty((0, len(coef)))
        if n > 0:
            pred[mask] = design_all @ coef

        segments[str(seg + 1)] = {
            "count": n,
            "intercept": float(coef[0]),
            "a_volume": float(coef[1]),
            "profile_offsets": {level: float(coef[2 + i]) for i, level in enumerate(profile_levels)},
            "profile_reference": profile_reference,
            "volume_min": float(np.min(x[mask])) if np.any(mask) else None,
            "volume_max": float(np.max(x[mask])) if np.any(mask) else None,
            "formula": "massa = intercept_segmen + a_segmen*volume_cm3 + offset_profile",
        }

    pred = np.maximum(pred, 0.0)
    return pred, {
        "formula": "massa = intercept_segmen + a_segmen*volume_cm3 + offset_profile",
        "breaks_volume_cm3": breaks,
        "profile_reference": profile_reference,
        "profile_levels": profile_levels,
        "segments": segments,
    }


def fit_robust(rows):
    volume = np.array([r["volume_cm3"] for r in rows], dtype=float)
    y = np.array([r["actual_weight_g"] for r in rows], dtype=float)
    mean = float(np.mean(volume))
    std = float(np.std(volume))
    if std <= 1e-12:
        std = 1.0
    z = (volume - mean) / std
    x = np.column_stack([np.ones_like(z), z])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    weights = np.ones_like(y)
    for _ in range(ROBUST_MAX_ITER):
        pred = x @ coef
        residual = y - pred
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        sigma = max(1.4826 * mad, 1e-9)
        u = np.abs(residual) / (HUBER_DELTA * sigma)
        weights_new = np.where(u <= 1.0, 1.0, 1.0 / u)
        wx = x * weights_new[:, None]
        coef_new, *_ = np.linalg.lstsq(wx, y * weights_new, rcond=None)
        if np.max(np.abs(coef_new - coef)) < ROBUST_TOL:
            coef = coef_new
            weights = weights_new
            break
        coef = coef_new
        weights = weights_new
    pred = np.maximum(x @ coef, 0.0)
    return pred, {
        "formula": "z = (volume_cm3-mean)/std; massa = intercept + b1*z",
        "intercept": float(coef[0]),
        "b1_z_volume": float(coef[1]),
        "volume_mean": mean,
        "volume_std": std,
        "huber_delta": HUBER_DELTA,
        "min_weight": float(np.min(weights)),
        "median_weight": float(np.median(weights)),
    }


def ridge_solve(x, y, alpha):
    penalty = np.eye(x.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + alpha * penalty, x.T @ y)


def fit_poly_ridge(rows):
    volume = np.array([r["volume_cm3"] for r in rows], dtype=float)
    y = np.array([r["actual_weight_g"] for r in rows], dtype=float)
    mean = float(np.mean(volume))
    std = float(np.std(volume))
    if std <= 1e-12:
        std = 1.0
    z = (volume - mean) / std
    x = np.column_stack([np.ones_like(z), z, z ** 2])
    coef = ridge_solve(x, y, RIDGE_ALPHA)
    pred = np.maximum(x @ coef, 0.0)
    return pred, {
        "formula": "z = (volume_cm3-mean)/std; massa = intercept + b1*z + b2*z^2",
        "ridge_alpha": RIDGE_ALPHA,
        "volume_mean": mean,
        "volume_std": std,
        "intercept": float(coef[0]),
        "b1_z_volume": float(coef[1]),
        "b2_z_volume_squared": float(coef[2]),
    }


def fit_spline_ridge(rows):
    volume = np.array([r["volume_cm3"] for r in rows], dtype=float)
    y = np.array([r["actual_weight_g"] for r in rows], dtype=float)
    mean = float(np.mean(volume))
    std = float(np.std(volume))
    if std <= 1e-12:
        std = 1.0
    z = (volume - mean) / std
    knots_volume = sorted(set(float(np.quantile(volume, q)) for q in SPLINE_KNOT_QUANTILES))
    knots_z = [(k - mean) / std for k in knots_volume]
    cols = [np.ones_like(z), z]
    for knot in knots_z:
        cols.append(np.maximum(0.0, z - knot))
    x = np.column_stack(cols)
    coef = ridge_solve(x, y, RIDGE_ALPHA)
    pred = np.maximum(x @ coef, 0.0)
    return pred, {
        "formula": "massa = intercept + b1*z + sum(h_i*max(0,z-k_i))",
        "ridge_alpha": RIDGE_ALPHA,
        "volume_mean": mean,
        "volume_std": std,
        "knots_volume_cm3": knots_volume,
        "knots_z": [float(v) for v in knots_z],
        "coefficients": [float(v) for v in coef],
    }


def fit_method(method_key, rows, baseline_key):
    if method_key == "linear":
        return fit_linear(rows)
    if method_key == "poly2":
        return fit_poly2(rows)
    if method_key == "density_empirical":
        return fit_density_empirical(rows)
    if method_key == "density_manual":
        return fit_density_manual(rows, baseline_key)
    if method_key == "multiple":
        return fit_multiple(rows, include_profile=False)
    if method_key == "multiple_profile":
        return fit_multiple(rows, include_profile=True)
    if method_key == "piecewise":
        return fit_piecewise(rows)
    if method_key == "piecewise_profile":
        return fit_piecewise_profile(rows)
    if method_key == "robust":
        return fit_robust(rows)
    if method_key == "poly_ridge":
        return fit_poly_ridge(rows)
    if method_key == "spline_ridge":
        return fit_spline_ridge(rows)
    raise ValueError(f"Metode tidak dikenal: {method_key}")


def predict_piecewise_from_model(rows, model_info, include_profile=False):
    breaks = [float(v) for v in model_info.get("breaks_volume_cm3", [])]
    segments = model_info.get("segments", {})
    profile_levels = model_info.get("profile_levels", [])
    pred = []
    for row in rows:
        volume = float(row["volume_cm3"])
        seg_key = str(segment_index(volume, breaks) + 1)
        seg = segments.get(seg_key)
        if not seg and segments:
            seg = segments[sorted(segments.keys())[0]]
        if not seg:
            pred.append(0.0)
            continue
        a_val = float(seg.get("a", seg.get("a_volume", 0.0)))
        b_val = float(seg.get("b", seg.get("intercept", 0.0)))
        value = b_val + a_val * volume
        if include_profile:
            offsets = seg.get("profile_offsets", {})
            profile = row.get("profile_norm", "unknown")
            if profile in profile_levels:
                value += float(offsets.get(profile, 0.0))
        pred.append(max(value, 0.0))
    return np.array(pred, dtype=float)


def predict_multiple_from_model(rows, model_info):
    features = model_info.get("numeric_features", ENHANCED_NUMERIC_FEATURES)
    stats = model_info.get("numeric_standardization", {})
    coefficients = model_info.get("coefficients", {})
    x_num, _ = numeric_matrix(rows, features, existing_stats=stats)
    pred = np.full(len(rows), float(coefficients.get("intercept", 0.0)), dtype=float)
    for i, feature in enumerate(features):
        pred += float(coefficients.get(feature, 0.0)) * x_num[:, i]
    for level in model_info.get("profile_levels", []):
        pred += float(coefficients.get(f"profile_{level}", 0.0)) * np.array(
            [1.0 if row.get("profile_norm") == level else 0.0 for row in rows],
            dtype=float,
        )
    return np.maximum(pred, 0.0)


def predict_method(method_key, rows, baseline_key, model_info):
    x = np.array([r["volume_cm3"] for r in rows], dtype=float)
    if method_key == "density_manual":
        density = float(model_info.get("density_g_per_cm3", MANUAL_DENSITY_G_PER_CM3))
        return np.maximum(density * x, 0.0)
    if method_key == "density_empirical":
        density = float(model_info.get("density_g_per_cm3", 0.0))
        return np.maximum(density * x, 0.0)
    if method_key == "linear":
        return np.maximum(float(model_info["a"]) * x + float(model_info["b"]), 0.0)
    if method_key == "piecewise":
        return predict_piecewise_from_model(rows, model_info, include_profile=False)
    if method_key == "piecewise_profile":
        return predict_piecewise_from_model(rows, model_info, include_profile=True)
    if method_key in ["multiple", "multiple_profile"]:
        return predict_multiple_from_model(rows, model_info)
    raise ValueError(f"Prediksi data uji belum tersedia untuk metode: {method_key}")


def split_train_test_rows(rows):
    rows_sorted = sorted(rows, key=lambda r: (float(r["actual_weight_g"]), str(r.get("profile_norm", "")), str(r.get("sample_id", ""))))
    if len(rows_sorted) < 5:
        return rows_sorted, []
    train_rows = []
    test_rows = []
    for idx, row in enumerate(rows_sorted):
        if idx % int(round(1.0 / (1.0 - TRAIN_RATIO))) == int(round(1.0 / (1.0 - TRAIN_RATIO))) - 1:
            test_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, test_rows


def save_train_test_evaluation(groups):
    ensure_dir(SUMMARY_DIR)
    metric_rows = []
    pred_rows = []

    for folder_name, method_label, method_key in METHODS:
        y_all = []
        pred_all = []
        train_total = 0
        test_total = 0
        for baseline_key, rows in sorted(groups.items(), key=lambda kv: baseline_sort_key(kv[0])):
            train_rows, test_rows = split_train_test_rows(rows)
            train_total += len(train_rows)
            test_total += len(test_rows)
            if not test_rows or len(train_rows) < 2:
                continue
            try:
                _, model_info = fit_method(method_key, train_rows, baseline_key)
                pred_test = predict_method(method_key, test_rows, baseline_key, model_info)
            except Exception as exc:
                pred_rows.append({
                    "method_label": method_label,
                    "baseline_cm": baseline_key,
                    "sample_id": "",
                    "status": "error",
                    "catatan": str(exc),
                })
                continue

            y_test = np.array([r["actual_weight_g"] for r in test_rows], dtype=float)
            y_all.extend(y_test.tolist())
            pred_all.extend(pred_test.tolist())
            for row, pred_val in zip(test_rows, pred_test):
                actual = float(row["actual_weight_g"])
                abs_err = abs(float(pred_val) - actual)
                pred_rows.append({
                    "method_folder": folder_name,
                    "method_label": method_label,
                    "baseline_cm": row.get("baseline_cm", baseline_key),
                    "sample_id": row.get("sample_id", ""),
                    "profile": row.get("profile", ""),
                    "actual_weight_g": actual,
                    "pred_weight_g": float(pred_val),
                    "abs_error_g": abs_err,
                    "error_percent": "" if actual <= 0 else abs_err / actual * 100.0,
                    "volume_cm3": row.get("volume_cm3", ""),
                    "mean_height_cm": row.get("mean_height_cm", ""),
                    "median_height_cm": row.get("median_height_cm", ""),
                    "max_height_cm": row.get("max_height_cm", ""),
                    "p95_height_cm": row.get("p95_height_cm", ""),
                    "p98_height_cm": row.get("p98_height_cm", ""),
                    "valid_pixels": row.get("valid_pixels", ""),
                    "area_batubara_cm2": row.get("area_batubara_cm2", ""),
                    "status": "uji_20_persen",
                    "catatan": "data ini tidak dipakai saat fitting model",
                })

        if y_all:
            metrics = compute_metrics(y_all, pred_all)
            metric_rows.append({
                "method_folder": folder_name,
                "method_label": method_label,
                "train_ratio": TRAIN_RATIO,
                "train_count": train_total,
                "test_count": test_total,
                "mae_test_gram": metrics["mae_gram"],
                "rmse_test_gram": metrics["rmse_gram"],
                "mape_test_percent": "" if metrics["mape_percent"] is None else metrics["mape_percent"],
                "r2_test": metrics["r2"],
                "high_error_count_test": count_high_errors(y_all, pred_all, metrics["mae_gram"]),
            })

    metric_rows.sort(key=lambda r: (float(r["mae_test_gram"]), float(r["rmse_test_gram"])))
    for idx, row in enumerate(metric_rows, start=1):
        row["ranking_test"] = idx

    metrics_csv = os.path.join(SUMMARY_DIR, "00_train_test_80_20_metrik.csv")
    predictions_csv = os.path.join(SUMMARY_DIR, "00_train_test_80_20_prediksi_data_uji.csv")
    summary_txt = os.path.join(SUMMARY_DIR, "00_train_test_80_20_penjelasan.txt")
    if metric_rows:
        write_csv(metrics_csv, metric_rows, list(metric_rows[0].keys()))
    if pred_rows:
        write_csv(predictions_csv, pred_rows, list(pred_rows[0].keys()))
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("EVALUASI TRAIN-TEST 80:20\n")
        f.write("=========================\n")
        f.write(f"Train ratio : {TRAIN_RATIO:.2f}\n")
        f.write(f"Random seed : {RANDOM_SEED}\n\n")
        f.write("Maksud evaluasi ini adalah menguji model pada data yang tidak dipakai saat proses fitting.\n")
        f.write("Data diurutkan dari massa kecil sampai besar, lalu sebagian data dipisahkan sebagai data uji.\n")
        f.write("Jika error data uji tetap kecil, model lebih layak dipakai untuk prediksi sampel baru.\n\n")
        for row in metric_rows:
            f.write(
                f"{row['ranking_test']}. {row['method_label']} | "
                f"MAE test {float(row['mae_test_gram']):.3f} g | "
                f"RMSE test {float(row['rmse_test_gram']):.3f} g | "
                f"MAPE test {float(row['mape_test_percent']):.3f}% | "
                f"R2 test {float(row['r2_test']):.6f}\n"
            )
    return {
        "train_test_metrics_csv": metrics_csv,
        "train_test_predictions_csv": predictions_csv,
        "train_test_summary_txt": summary_txt,
        "train_test_rows": metric_rows,
    }


# =========================================================
# OUTPUT PER METODE
# =========================================================
def prediction_rows_for_method(groups, folder_name, method_label, method_key):
    models = {}
    pred_rows = []
    for baseline_key, rows in sorted(groups.items(), key=lambda kv: baseline_sort_key(kv[0])):
        try:
            pred, model_info = fit_method(method_key, rows, baseline_key)
        except Exception as exc:
            models[str(baseline_key)] = {"error": str(exc), "jumlah_data": len(rows)}
            continue

        y = np.array([r["actual_weight_g"] for r in rows], dtype=float)
        metrics = compute_metrics(y, pred)
        models[str(baseline_key)] = {
            "method_label": method_label,
            "method_key": method_key,
            "jumlah_data": len(rows),
            "model": model_info,
            "mae_gram": metrics["mae_gram"],
            "rmse_gram": metrics["rmse_gram"],
            "mape_percent": metrics["mape_percent"],
            "r2": metrics["r2"],
        }

        for idx, row in enumerate(rows):
            actual = float(row["actual_weight_g"])
            pred_val = float(pred[idx])
            abs_err = abs(pred_val - actual)
            err_percent = "" if actual <= 0 else abs_err / actual * 100.0
            pred_rows.append({
                "method_folder": folder_name,
                "method_label": method_label,
                "method_key": method_key,
                "baseline_cm": row.get("baseline_cm", baseline_key),
                "baseline_key": baseline_key,
                "sample_id": row.get("sample_id", ""),
                "target_massa_gram": "" if row.get("target_weight_g") is None else row.get("target_weight_g"),
                "massa_aktual_gram": actual,
                "massa_prediksi_gram": pred_val,
                "selisih_gram": pred_val - actual,
                "abs_selisih_gram": abs_err,
                "error_persen": err_percent,
                "volume_cm3": row.get("volume_cm3", ""),
                "mean_height_cm": "" if row.get("mean_height_cm") is None else row.get("mean_height_cm"),
                "median_height_cm": "" if row.get("median_height_cm") is None else row.get("median_height_cm"),
                "max_height_cm": "" if row.get("max_height_cm") is None else row.get("max_height_cm"),
                "p95_height_cm": "" if row.get("p95_height_cm") is None else row.get("p95_height_cm"),
                "p98_height_cm": "" if row.get("p98_height_cm") is None else row.get("p98_height_cm"),
                "valid_pixels": "" if row.get("valid_pixels") is None else row.get("valid_pixels"),
                "area_batubara_cm2": "" if row.get("area_batubara_cm2") is None else row.get("area_batubara_cm2"),
                "condition": row.get("condition", ""),
                "profile": row.get("profile", ""),
                "repetition": row.get("repetition", ""),
                "status": row.get("status", ""),
                "catatan": row.get("catatan", ""),
                "_actual_weight_g": actual,
                "_pred_weight_g": pred_val,
            })
    return models, pred_rows


def public_prediction_rows(pred_rows):
    skip = {"_actual_weight_g", "_pred_weight_g"}
    return [{k: v for k, v in row.items() if k not in skip} for row in pred_rows]


def save_method_outputs(folder_name, method_label, method_key, groups):
    method_dir = os.path.join(OUTPUT_ROOT, folder_name)
    ensure_dir(method_dir)
    models, pred_rows = prediction_rows_for_method(groups, folder_name, method_label, method_key)
    public_rows = public_prediction_rows(pred_rows)

    result_csv = os.path.join(method_dir, f"{folder_name}_hasil_prediksi.csv")
    if public_rows:
        write_csv(result_csv, public_rows, list(public_rows[0].keys()))

    metrics = aggregate_method_metrics([
        {"actual_weight_g": r["_actual_weight_g"], "pred_weight_g": r["_pred_weight_g"]}
        for r in pred_rows
    ]) if pred_rows else None

    model_json = os.path.join(method_dir, f"{folder_name}_model_dan_metrik.json")
    with open(model_json, "w", encoding="utf-8") as f:
        json.dump({
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_csv": CALIBRATION_CSV,
            "method_label": method_label,
            "method_key": method_key,
            "method_detail": METHOD_DETAILS.get(method_key, {}),
            "overall_metrics": metrics,
            "models_by_baseline_cm": models,
        }, f, indent=2)

    summary_txt = os.path.join(method_dir, f"{folder_name}_ringkasan.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"RINGKASAN METODE - {method_label.upper()}\n")
        f.write("=" * 72 + "\n")
        f.write(f"Waktu       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sumber data : {CALIBRATION_CSV}\n\n")
        if metrics is None:
            f.write("Tidak ada prediksi valid.\n")
        else:
            f.write(f"Jumlah data       : {metrics['jumlah_data']}\n")
            f.write(f"MAE               : {metrics['mae_gram']:.6f} gram\n")
            f.write(f"RMSE              : {metrics['rmse_gram']:.6f} gram\n")
            if metrics["mape_percent"] is None:
                f.write("MAPE              : tidak dihitung\n")
            else:
                f.write(f"MAPE              : {metrics['mape_percent']:.6f}%\n")
            f.write(f"R2                : {metrics['r2']:.6f}\n")
            f.write(f"Jumlah error tinggi: {metrics['high_error_count']}\n\n")
        f.write("Metrik per baseline\n")
        f.write("-------------------\n")
        for baseline_key, model in sorted(models.items(), key=lambda kv: baseline_sort_key(kv[0])):
            f.write(f"Baseline {baseline_key} cm\n")
            if "error" in model:
                f.write(f"  ERROR: {model['error']}\n")
            else:
                f.write(f"  Data : {model['jumlah_data']}\n")
                f.write(f"  MAE  : {model['mae_gram']:.6f} gram\n")
                f.write(f"  RMSE : {model['rmse_gram']:.6f} gram\n")
                f.write(f"  MAPE : {model['mape_percent'] if model['mape_percent'] is not None else 'N/A'}\n")
                f.write(f"  R2   : {model['r2']:.6f}\n")
            f.write("\n")

    graph_path = save_method_scatter(method_dir, folder_name, method_label, method_key, pred_rows, models)
    actual_prediction_path = save_method_actual_prediction_graph(method_dir, folder_name, method_label, pred_rows)
    bar_paths = save_method_bar_charts(method_dir, folder_name, method_label, pred_rows)

    return {
        "folder_name": folder_name,
        "method_label": method_label,
        "method_key": method_key,
        "method_dir": method_dir,
        "result_csv": result_csv,
        "model_json": model_json,
        "summary_txt": summary_txt,
        "graph_path": graph_path,
        "actual_prediction_path": actual_prediction_path,
        "bar_paths": bar_paths,
        "piecewise_paths": [],
        "metrics": metrics,
        "pred_rows": pred_rows,
        "models": models,
    }


# =========================================================
# GRAFIK
# =========================================================
def plot_volume_model_graph(ax, method_label, method_key, pred_rows, models, rank_row=None):
    by_baseline = defaultdict(list)
    for row in pred_rows:
        by_baseline[row["baseline_key"]].append(row)

    has_error_label = False
    has_actual_label = False
    has_pred_label = False
    segment_names = ["rendah", "sedang", "tinggi", "sangat tinggi"]
    segment_colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    for baseline_key, rows in sorted(by_baseline.items(), key=lambda kv: baseline_sort_key(kv[0])):
        rows_sorted = sorted(rows, key=lambda r: parse_float(r.get("volume_cm3"), 0.0))
        volume = np.array([parse_float(r.get("volume_cm3"), 0.0) for r in rows_sorted], dtype=float)
        actual = np.array([r["_actual_weight_g"] for r in rows_sorted], dtype=float)
        pred = np.array([r["_pred_weight_g"] for r in rows_sorted], dtype=float)

        ax.scatter(
            volume,
            actual,
            s=22,
            color="#1f77b4",
            label="Titik biru = data aktual" if not has_actual_label else None,
            zorder=3,
        )
        has_actual_label = True

        for xi, yi, pi in zip(volume, actual, pred):
            ax.plot(
                [xi, xi],
                [yi, pi],
                linestyle="--",
                color="#999999",
                linewidth=0.8,
                alpha=0.65,
                label="Garis abu putus = selisih error" if not has_error_label else None,
            )
            has_error_label = True

        if method_key in ["piecewise", "piecewise_profile"]:
            model = models.get(str(baseline_key), {})
            info = model.get("model", {})
            breaks = [float(v) for v in info.get("breaks_volume_cm3", [])]
            segments = info.get("segments", {})
            if len(volume) == 0:
                continue
            boundaries = [float(np.min(volume))] + breaks + [float(np.max(volume))]
            for idx in range(len(boundaries) - 1):
                seg = segments.get(str(idx + 1), {})
                a_val = seg.get("a", seg.get("a_volume"))
                b_val = seg.get("b", seg.get("intercept"))
                if a_val is None or b_val is None:
                    continue
                x0, x1 = boundaries[idx], boundaries[idx + 1]
                x_line = np.linspace(x0, x1, 50)
                y_line = np.maximum(float(a_val) * x_line + float(b_val), 0.0)
                label = f"Garis model segmen {segment_names[idx] if idx < len(segment_names) else idx + 1}"
                ax.plot(
                    x_line,
                    y_line,
                    color=segment_colors[idx % len(segment_colors)],
                    linewidth=2.0,
                    label=label,
                    zorder=2,
                )
            for br in breaks:
                ax.axvline(br, linestyle=":", color="#555555", linewidth=1.1)
                ymax = ax.get_ylim()[1]
                ax.text(br, ymax * 0.96, "batas\nsegmen", ha="center", va="top", fontsize=7, color="#444444")
        else:
            ax.plot(
                volume,
                pred,
                color="#ff7f0e",
                linewidth=1.8,
                marker="o",
                markersize=3,
                label="Garis oranye = hasil model" if not has_pred_label else None,
                zorder=2,
            )
            has_pred_label = True

    title = method_label
    if rank_row:
        title += (
            f" | MAE {float(rank_row.get('mae_gram', 0)):.1f} g"
            f" | MAPE {float(rank_row.get('mape_percent', 0)):.1f}%"
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Volume hasil sistem (cm3)", fontsize=8)
    ax.set_ylabel("Massa (gram)", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6, loc="best")

    if method_key in ["piecewise", "piecewise_profile"]:
        note = "Piecewise: model dibagi 3 segmen volume"
    else:
        note = "Titik biru aktual, garis oranye model"
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )


def save_method_scatter(method_dir, folder_name, method_label, method_key, pred_rows, models):
    if not MATPLOTLIB_OK or not pred_rows:
        return ""
    path = os.path.join(method_dir, f"{folder_name}_grafik_volume_ke_massa.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_volume_model_graph(ax, method_label, method_key, pred_rows, models)
    plt.figtext(
        0.5,
        0.01,
        "Keterangan: titik biru adalah massa aktual dari timbangan; garis model menunjukkan hasil estimasi sistem; "
        "garis abu putus menunjukkan jarak error antara aktual dan prediksi.",
        ha="center",
        fontsize=8,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_actual_prediction_graph(ax, method_label, pred_rows, rank_row=None):
    actual = np.array([r["_actual_weight_g"] for r in pred_rows], dtype=float)
    pred = np.array([r["_pred_weight_g"] for r in pred_rows], dtype=float)
    max_val = max(float(np.max(actual)), float(np.max(pred)), 1.0)
    ax.scatter(actual, pred, s=22, color="#1f77b4", alpha=0.8, label="Titik biru = data prediksi")
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="#444444", linewidth=1.0, label="Garis putus = prediksi ideal")
    title = method_label
    if rank_row:
        title += (
            f" | MAE {float(rank_row.get('mae_gram', 0)):.1f} g"
            f" | MAPE {float(rank_row.get('mape_percent', 0)):.1f}%"
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Massa aktual dari timbangan (gram)", fontsize=8)
    ax.set_ylabel("Massa prediksi sistem (gram)", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6, loc="best")
    ax.text(
        0.02,
        0.02,
        "Dekat garis = akurat\nJauh dari garis = error besar",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )


def save_method_actual_prediction_graph(method_dir, folder_name, method_label, pred_rows):
    if not MATPLOTLIB_OK or not pred_rows:
        return ""
    path = os.path.join(method_dir, f"{folder_name}_grafik_aktual_vs_prediksi.png")
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_actual_prediction_graph(ax, method_label, pred_rows)
    plt.figtext(
        0.5,
        0.01,
        "Keterangan: titik makin dekat ke garis putus-putus berarti massa prediksi sistem makin dekat dengan massa aktual.",
        ha="center",
        fontsize=8,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_method_bar_charts(method_dir, folder_name, method_label, pred_rows):
    if not MATPLOTLIB_OK or not pred_rows:
        return []
    paths = []
    by_baseline = defaultdict(list)
    for row in pred_rows:
        by_baseline[row["baseline_key"]].append(row)
    for baseline_key, rows in sorted(by_baseline.items(), key=lambda kv: baseline_sort_key(kv[0])):
        chart_rows = sorted(rows, key=lambda r: r["_actual_weight_g"])
        labels = [str(r["sample_id"]) for r in chart_rows]
        actual = [r["_actual_weight_g"] for r in chart_rows]
        pred = [r["_pred_weight_g"] for r in chart_rows]
        errors = [abs(p - a) for a, p in zip(actual, pred)]
        y_pos = np.arange(len(labels))
        fig_height = max(6.0, len(labels) * 0.34)
        bar_height = 0.38
        path = os.path.join(method_dir, f"{folder_name}_diagram_batang_baseline_{baseline_key}cm.png")
        plt.figure(figsize=(13, fig_height))
        plt.barh(y_pos - bar_height / 2, actual, height=bar_height, label="Massa aktual")
        plt.barh(y_pos + bar_height / 2, pred, height=bar_height, label="Hasil sistem")
        plt.yticks(y_pos, labels, fontsize=8)
        plt.gca().invert_yaxis()
        plt.xlabel("Massa (gram)")
        plt.title(f"{method_label} - Baseline {baseline_key} cm")
        plt.grid(axis="x", alpha=0.3)
        plt.legend()
        plt.figtext(
            0.5,
            0.01,
            "Keterangan: batang biru = massa aktual dari timbangan; batang oranye = estimasi sistem; "
            "teks error di kanan = selisih absolut antara aktual dan sistem. Data diurutkan dari massa aktual paling ringan ke paling berat.",
            ha="center",
            fontsize=8,
        )
        max_value = max(actual + pred) if actual or pred else 0.0
        label_x = max_value * 1.01 if max_value > 0 else 1.0
        for y_idx, err in zip(y_pos, errors):
            plt.text(label_x, y_idx, f"error {err:.1f} g", va="center", fontsize=7)
        plt.tight_layout(rect=[0, 0.035, 1, 1])
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)
    return paths


def save_piecewise_volume_graphs(method_dir, folder_name, method_label, pred_rows, models):
    if not MATPLOTLIB_OK or not pred_rows:
        return []
    paths = []
    by_baseline = defaultdict(list)
    for row in pred_rows:
        by_baseline[row["baseline_key"]].append(row)

    segment_names = ["rendah", "sedang", "tinggi", "sangat tinggi"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    for baseline_key, rows in sorted(by_baseline.items(), key=lambda kv: baseline_sort_key(kv[0])):
        model = models.get(str(baseline_key), {})
        if "model" not in model:
            continue
        info = model["model"]
        breaks = [float(v) for v in info.get("breaks_volume_cm3", [])]
        segments = info.get("segments", {})
        volume = np.array([parse_float(r.get("volume_cm3"), 0.0) for r in rows], dtype=float)
        actual = np.array([r["_actual_weight_g"] for r in rows], dtype=float)
        pred = np.array([r["_pred_weight_g"] for r in rows], dtype=float)
        if len(volume) == 0:
            continue

        path = os.path.join(method_dir, f"{folder_name}_grafik_piecewise_3_segmen_baseline_{baseline_key}cm.png")
        plt.figure(figsize=(12, 7))
        plt.scatter(volume, actual, color="#333333", label="Data aktual", zorder=3)
        for xi, yi, pi in zip(volume, actual, pred):
            plt.plot([xi, xi], [yi, pi], linestyle="--", color="#999999", linewidth=0.8, alpha=0.7)

        boundaries = [float(np.min(volume))] + breaks + [float(np.max(volume))]
        for idx in range(len(boundaries) - 1):
            seg = segments.get(str(idx + 1), {})
            if "a" not in seg or "b" not in seg:
                continue
            x0, x1 = boundaries[idx], boundaries[idx + 1]
            if idx > 0:
                x0 = max(x0, boundaries[idx])
            x_line = np.linspace(x0, x1, 60)
            y_line = np.maximum(float(seg["a"]) * x_line + float(seg["b"]), 0.0)
            label = f"Segmen {segment_names[idx] if idx < len(segment_names) else idx + 1}"
            plt.plot(x_line, y_line, color=colors[idx % len(colors)], linewidth=2.0, label=label)

        for br in breaks:
            plt.axvline(br, linestyle=":", color="#666666", linewidth=1.2)
            plt.text(br, plt.ylim()[1] * 0.96, "batas\nsegmen", ha="center", va="top", fontsize=8, color="#555555")

        plt.xlabel("Volume hasil sistem (cm3)")
        plt.ylabel("Massa aktual / prediksi (gram)")
        plt.title(f"Piecewise Linear Volume ke Massa - Baseline {baseline_key} cm")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.figtext(
            0.5,
            0.01,
            "Keterangan: piecewise membagi volume menjadi 3 rentang. Setiap warna adalah persamaan linear berbeda "
            "untuk segmen rendah, sedang, dan tinggi; garis vertikal putus-putus adalah batas segmen.",
            ha="center",
            fontsize=8,
        )
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)
    return paths


def save_summary_graphs(method_results, ranking_rows):
    if not MATPLOTLIB_OK:
        return "", "", ""

    order_by_folder = {row["method_folder"]: row for row in ranking_rows}
    ordered_results = sorted(
        [res for res in method_results if res["folder_name"] in order_by_folder],
        key=lambda res: int(order_by_folder[res["folder_name"]]["ranking_akhir"]),
        reverse=True,
    )

    combined_path = os.path.join(SUMMARY_DIR, "00_gabungan_model_grafik_volume_ke_massa.png")
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    axes = axes.flatten()
    for ax, res in zip(axes, ordered_results):
        rows = res["pred_rows"]
        rank_row = order_by_folder.get(res["folder_name"], {})
        if not rows:
            ax.set_title(res["method_label"])
            ax.text(0.5, 0.5, "Tidak ada data", ha="center", va="center")
            ax.axis("off")
            continue
        plot_volume_model_graph(ax, res["method_label"], res["method_key"], rows, res["models"], rank_row)
    for ax in axes[len(ordered_results):]:
        ax.axis("off")
    fig.text(
        0.5,
        0.01,
        "Urutan grafik: dari metode dengan error/ranking terburuk menuju metode terbaik. "
        "Titik biru = data aktual, garis/marker model = hasil sistem, garis abu putus = besar error. "
        "Pada piecewise, garis model terbagi menjadi segmen rendah, sedang, dan tinggi.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.035)
    plt.savefig(combined_path, dpi=180)
    plt.close()

    actual_combined_path = os.path.join(SUMMARY_DIR, "00_gabungan_model_grafik_aktual_vs_prediksi.png")
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    axes = axes.flatten()
    for ax, res in zip(axes, ordered_results):
        rows = res["pred_rows"]
        rank_row = order_by_folder.get(res["folder_name"], {})
        if not rows:
            ax.set_title(res["method_label"])
            ax.text(0.5, 0.5, "Tidak ada data", ha="center", va="center")
            ax.axis("off")
            continue
        plot_actual_prediction_graph(ax, res["method_label"], rows, rank_row)
    for ax in axes[len(ordered_results):]:
        ax.axis("off")
    fig.text(
        0.5,
        0.01,
        "Urutan grafik: dari metode dengan error/ranking terburuk menuju metode terbaik. "
        "Titik biru = prediksi tiap data; garis putus-putus = prediksi ideal saat hasil sistem sama dengan massa aktual.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.035)
    plt.savefig(actual_combined_path, dpi=180)
    plt.close()

    metric_path = os.path.join(SUMMARY_DIR, "00_ranking_mae_rmse_mape.png")
    valid_rows = [r for r in ranking_rows if r.get("mape_percent") not in ["", None]]
    valid_rows = sorted(valid_rows, key=lambda r: int(r["ranking_akhir"]), reverse=True)
    labels = [r["method_label"] for r in valid_rows]
    mae = [float(r["mae_gram"]) for r in valid_rows]
    rmse = [float(r["rmse_gram"]) for r in valid_rows]
    mape = [float(r["mape_percent"]) for r in valid_rows]
    x = np.arange(len(labels))
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].bar(x - width / 2, mae, width=width, label="MAE")
    axes[0].bar(x + width / 2, rmse, width=width, label="RMSE")
    axes[0].set_ylabel("Error (gram)")
    axes[0].set_title("Perbandingan Error Gram dan Error Persen 5 Model (Terburuk ke Terbaik)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()
    axes[1].bar(x, mape, width=0.55, label="MAPE", color="#2ca02c")
    axes[1].set_ylabel("MAPE (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    fig.text(
        0.5,
        0.01,
        "Keterangan: MAE/RMSE menunjukkan selisih dalam gram; MAPE menunjukkan selisih relatif dalam persen terhadap massa aktual.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(metric_path, dpi=180)
    plt.close()
    return combined_path, actual_combined_path, metric_path


# =========================================================
# RANKING DAN ANALISIS
# =========================================================
def rank_values(values, reverse=False):
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1], reverse=reverse)
    ranks = [0] * len(values)
    for rank, (idx, value) in enumerate(indexed, start=1):
        ranks[idx] = rank
    return ranks


def build_ranking(method_results):
    valid = [res for res in method_results if res["metrics"] is not None]
    mae = [res["metrics"]["mae_gram"] for res in valid]
    rmse = [res["metrics"]["rmse_gram"] for res in valid]
    mape = [res["metrics"]["mape_percent"] if res["metrics"]["mape_percent"] is not None else 1e12 for res in valid]
    r2 = [res["metrics"]["r2"] for res in valid]
    high = [res["metrics"]["high_error_count"] for res in valid]

    rank_mae = rank_values(mae)
    rank_rmse = rank_values(rmse)
    rank_mape = rank_values(mape)
    rank_r2 = rank_values(r2, reverse=True)
    rank_high = rank_values(high)

    rows = []
    for i, res in enumerate(valid):
        score = rank_mae[i] + rank_rmse[i] + rank_mape[i] + rank_r2[i] + rank_high[i]
        metrics = res["metrics"]
        rows.append({
            "ranking_score": score,
            "rank_mae": rank_mae[i],
            "rank_rmse": rank_rmse[i],
            "rank_mape": rank_mape[i],
            "rank_r2": rank_r2[i],
            "rank_high_error": rank_high[i],
            "method_folder": res["folder_name"],
            "method_label": res["method_label"],
            "jumlah_data": metrics["jumlah_data"],
            "mae_gram": metrics["mae_gram"],
            "rmse_gram": metrics["rmse_gram"],
            "mape_percent": "" if metrics["mape_percent"] is None else metrics["mape_percent"],
            "r2": metrics["r2"],
            "high_error_count": metrics["high_error_count"],
        })
    rows.sort(key=lambda r: (r["ranking_score"], r["mae_gram"], r["rmse_gram"]))
    for idx, row in enumerate(rows, start=1):
        row["ranking_akhir"] = idx
    return rows


def scoring_explanation_lines():
    return [
        "Cara pemberian score:",
        "1. Setiap metode diberi peringkat untuk MAE, RMSE, MAPE, R2, dan jumlah error tinggi.",
        "2. Untuk MAE, RMSE, MAPE, dan jumlah error tinggi, nilai paling kecil mendapat rank 1 karena error lebih rendah berarti lebih baik.",
        "3. Untuk R2, nilai paling besar mendapat rank 1 karena semakin mendekati 1 berarti prediksi lebih sesuai dengan data aktual.",
        "4. ranking_score = rank_MAE + rank_RMSE + rank_MAPE + rank_R2 + rank_error_tinggi.",
        "5. Metode terbaik adalah metode dengan ranking_score paling kecil. Jika score sama, script memprioritaskan MAE dan RMSE yang lebih kecil.",
    ]


def method_explanation_rows():
    rows = []
    for folder_name, method_label, method_key in METHODS:
        detail = METHOD_DETAILS.get(method_key, {})
        rows.append({
            "method_folder": folder_name,
            "method_label": method_label,
            "method_key": method_key,
            "rumus": detail.get("rumus", ""),
            "penjelasan_metode": detail.get("penjelasan", ""),
            "alasan_relevan": detail.get("relevansi", ""),
        })
    return rows


def worst_to_best_rows(ranking_rows):
    return sorted(ranking_rows, key=lambda r: int(r["ranking_akhir"]), reverse=True)


def create_ordered_folder_copies(method_results, ranking_rows):
    ordered_root = os.path.join(OUTPUT_ROOT, "01_urutan_folder_error_tertinggi_ke_terbaik")
    if os.path.exists(ordered_root):
        shutil.rmtree(ordered_root)
    ensure_dir(ordered_root)

    result_by_folder = {res["folder_name"]: res for res in method_results}
    ordered_rows = worst_to_best_rows(ranking_rows)
    index_rows = []
    for idx, row in enumerate(ordered_rows, start=1):
        res = result_by_folder.get(row["method_folder"])
        if res is None:
            continue
        dest_name = f"{idx:02d}_rank_{row['ranking_akhir']}_{row['method_folder']}"
        dest_dir = os.path.join(ordered_root, dest_name)
        shutil.copytree(res["method_dir"], dest_dir)
        index_rows.append({
            "urutan_error_tertinggi_ke_terbaik": idx,
            "ranking_akhir": row["ranking_akhir"],
            "folder_urut": dest_name,
            "folder_asli": row["method_folder"],
            "method_label": row["method_label"],
            "ranking_score": row["ranking_score"],
            "mae_gram": row["mae_gram"],
            "rmse_gram": row["rmse_gram"],
            "mape_percent": row["mape_percent"],
            "r2": row["r2"],
            "high_error_count": row["high_error_count"],
        })
    if index_rows:
        write_csv(
            os.path.join(ordered_root, "00_index_urutan_folder.csv"),
            index_rows,
            list(index_rows[0].keys()),
        )
    return ordered_root


def sample_key(row):
    return (str(row.get("baseline_key", "")), str(row.get("sample_id", "")))


def build_recheck_rows(method_results, best_result):
    if best_result is None or best_result["metrics"] is None:
        return []

    best_mae = float(best_result["metrics"]["mae_gram"])
    threshold = max(ABS_ERROR_RECHECK_GRAM, 1.5 * best_mae)

    all_errors = defaultdict(list)
    row_meta = {}
    for res in method_results:
        for row in res["pred_rows"]:
            key = sample_key(row)
            actual = float(row["_actual_weight_g"])
            pred = float(row["_pred_weight_g"])
            abs_err = abs(pred - actual)
            pct = None if actual <= 0 else abs_err / actual * 100.0
            all_errors[key].append({
                "method_label": res["method_label"],
                "abs_err": abs_err,
                "pct_err": pct,
            })
            row_meta[key] = row

    best_errors = {}
    for row in best_result["pred_rows"]:
        key = sample_key(row)
        actual = float(row["_actual_weight_g"])
        pred = float(row["_pred_weight_g"])
        abs_err = abs(pred - actual)
        pct = None if actual <= 0 else abs_err / actual * 100.0
        best_errors[key] = (abs_err, pct, pred)

    out = []
    for key, errors in all_errors.items():
        if key not in best_errors:
            continue
        best_abs, best_pct, best_pred = best_errors[key]
        high_count = 0
        for err in errors:
            pct_bad = err["pct_err"] is not None and err["pct_err"] >= PERCENT_ERROR_RECHECK
            if err["abs_err"] >= threshold or pct_bad:
                high_count += 1
        pct_best_bad = best_pct is not None and best_pct >= PERCENT_ERROR_RECHECK
        if best_abs >= threshold or pct_best_bad or high_count >= max(3, math.ceil(len(method_results) * 0.4)):
            meta = row_meta[key]
            alasan = []
            if best_abs >= threshold:
                alasan.append(f"error metode terbaik >= {threshold:.1f} g")
            if pct_best_bad:
                alasan.append(f"error persen metode terbaik >= {PERCENT_ERROR_RECHECK:.1f}%")
            if high_count >= max(3, math.ceil(len(method_results) * 0.4)):
                alasan.append("error tinggi muncul di banyak metode")
            mean_abs = float(np.mean([e["abs_err"] for e in errors]))
            max_abs = float(np.max([e["abs_err"] for e in errors]))
            out.append({
                "baseline_cm": meta.get("baseline_cm", ""),
                "sample_id": meta.get("sample_id", ""),
                "profile": meta.get("profile", ""),
                "condition": meta.get("condition", ""),
                "actual_gram": meta.get("_actual_weight_g", ""),
                "prediksi_metode_terbaik_gram": best_pred,
                "abs_error_metode_terbaik_gram": best_abs,
                "error_persen_metode_terbaik": "" if best_pct is None else best_pct,
                "jumlah_metode_error_tinggi": high_count,
                "mean_abs_error_semua_metode_gram": mean_abs,
                "max_abs_error_semua_metode_gram": max_abs,
                "volume_cm3": meta.get("volume_cm3", ""),
                "mean_height_cm": meta.get("mean_height_cm", ""),
                "max_height_cm": meta.get("max_height_cm", ""),
                "valid_pixels": meta.get("valid_pixels", ""),
                "alasan_cek_ulang": "; ".join(alasan),
            })
    out.sort(key=lambda r: (float(r["abs_error_metode_terbaik_gram"]), float(r["mean_abs_error_semua_metode_gram"])), reverse=True)
    return out


def analysis_text(ranking_rows, recheck_rows):
    if not ranking_rows:
        return ["Belum ada metode yang berhasil dihitung."]
    best = ranking_rows[0]
    second = ranking_rows[1] if len(ranking_rows) > 1 else None
    lines = []
    lines.append(
        f"Metode terbaik berdasarkan ranking gabungan adalah {best['method_label']} "
        f"dengan MAE {float(best['mae_gram']):.3f} g, RMSE {float(best['rmse_gram']):.3f} g, "
        f"MAPE {float(best['mape_percent']):.3f}% dan R2 {float(best['r2']):.6f}."
    )
    if second is not None:
        lines.append(
            f"Pembanding terdekat adalah {second['method_label']} dengan MAE "
            f"{float(second['mae_gram']):.3f} g, RMSE {float(second['rmse_gram']):.3f} g, "
            f"MAPE {float(second['mape_percent']):.3f}% dan R2 {float(second['r2']):.6f}."
        )
    lines.append(
        "Pemilihan ini tidak hanya memakai R2, karena R2 tinggi dapat tetap menyisakan error besar "
        "pada muatan tertentu. MAE menunjukkan besar kesalahan rata-rata dalam gram, RMSE memberi "
        "hukuman lebih besar pada error ekstrem, MAPE menunjukkan kesalahan relatif terhadap massa "
        "aktual, sedangkan jumlah error tinggi menunjukkan kestabilan metode pada data sulit."
    )
    if recheck_rows:
        top = recheck_rows[0]
        lines.append(
            f"Data yang paling perlu dicek ulang adalah {top['sample_id']} pada baseline "
            f"{top['baseline_cm']} cm karena error metode terbaik mencapai "
            f"{float(top['abs_error_metode_terbaik_gram']):.3f} g. Data seperti ini perlu dicek "
            "pada sisi massa aktual, segmentasi mask, depth/height map, volume, dan posisi/bentuk muatan."
        )
        lines.append(
            f"Total data yang masuk daftar cek ulang adalah {len(recheck_rows)} sampel. "
            "Prioritas awal adalah sampel yang error-nya besar pada metode terbaik dan juga konsisten "
            "besar pada banyak metode, karena pola tersebut lebih mungkin berasal dari masalah data "
            "atau kondisi pengukuran, bukan hanya kelemahan satu model."
        )
    else:
        lines.append("Tidak ada data yang melewati ambang cek ulang berdasarkan aturan error tinggi saat ini.")
    return lines


def save_summary_files(method_results, ranking_rows, recheck_rows):
    ensure_dir(SUMMARY_DIR)
    ranking_csv = os.path.join(SUMMARY_DIR, "00_perbandingan_metrik_5_model.csv")
    if ranking_rows:
        write_csv(ranking_csv, ranking_rows, list(ranking_rows[0].keys()))

    worst_order_rows = worst_to_best_rows(ranking_rows)
    worst_order_csv = os.path.join(SUMMARY_DIR, "00_urutan_metode_error_tertinggi_ke_terbaik.csv")
    if worst_order_rows:
        write_csv(worst_order_csv, worst_order_rows, list(worst_order_rows[0].keys()))

    method_rows = method_explanation_rows()
    method_table_csv = os.path.join(SUMMARY_DIR, "00_tabel_penjelasan_5_model.csv")
    write_csv(method_table_csv, method_rows, list(method_rows[0].keys()))

    recheck_csv = os.path.join(SUMMARY_DIR, "00_data_perlu_cek_ulang_error_tinggi.csv")
    if recheck_rows:
        write_csv(recheck_csv, recheck_rows, list(recheck_rows[0].keys()))
    else:
        write_csv(recheck_csv, [], [
            "baseline_cm", "sample_id", "profile", "condition", "actual_gram",
            "prediksi_metode_terbaik_gram", "abs_error_metode_terbaik_gram",
            "error_persen_metode_terbaik", "jumlah_metode_error_tinggi",
            "mean_abs_error_semua_metode_gram", "max_abs_error_semua_metode_gram",
            "volume_cm3", "mean_height_cm", "max_height_cm", "valid_pixels", "alasan_cek_ulang"
        ])

    text_lines = analysis_text(ranking_rows, recheck_rows)
    summary_txt = os.path.join(SUMMARY_DIR, "00_analisis_metode_terbaik_dan_data_cek_ulang.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("ANALISIS PERBANDINGAN 5 MODEL ESTIMASI MASSA BATUBARA\n")
        f.write("=" * 72 + "\n")
        f.write(f"Waktu evaluasi : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sumber data    : {CALIBRATION_CSV}\n\n")
        f.write("CARA PEMBERIAN SCORE\n")
        f.write("--------------------\n")
        for line in scoring_explanation_lines():
            f.write(line + "\n")
        f.write("\n")
        f.write("PENJELASAN 5 MODEL\n")
        f.write("--------------------\n")
        for row in method_rows:
            f.write(f"{row['method_label']}\n")
            f.write(f"  Rumus       : {row['rumus']}\n")
            f.write(f"  Penjelasan  : {row['penjelasan_metode']}\n")
            f.write(f"  Relevansi   : {row['alasan_relevan']}\n\n")
        f.write("URUTAN METODE DARI ERROR TERTINGGI KE TERBAIK\n")
        f.write("---------------------------------------------\n")
        for row in worst_order_rows:
            f.write(
                f"{row['method_label']} | ranking akhir {row['ranking_akhir']} | "
                f"score {row['ranking_score']} | MAE {float(row['mae_gram']):.3f} g | "
                f"RMSE {float(row['rmse_gram']):.3f} g | MAPE {float(row['mape_percent']):.3f}% | "
                f"R2 {float(row['r2']):.6f}\n"
            )
        f.write("\n")
        f.write("TABEL METRIK METODE DARI ERROR TERTINGGI KE TERBAIK\n")
        f.write("---------------------------------------------------\n")
        for row in worst_order_rows:
            f.write(
                f"{row['ranking_akhir']}. {row['method_label']} | "
                f"score {row['ranking_score']} | MAE {float(row['mae_gram']):.3f} g | "
                f"RMSE {float(row['rmse_gram']):.3f} g | MAPE {float(row['mape_percent']):.3f}% | "
                f"R2 {float(row['r2']):.6f} | error tinggi {row['high_error_count']}\n"
            )
        f.write("\nANALISIS\n")
        f.write("--------\n")
        for line in text_lines:
            f.write(line + "\n\n")
        f.write("DATA PRIORITAS CEK ULANG\n")
        f.write("------------------------\n")
        for row in recheck_rows[:20]:
            f.write(
                f"- {row['sample_id']} | baseline {row['baseline_cm']} cm | "
                f"profil {row['profile']} | aktual {float(row['actual_gram']):.3f} g | "
                f"error terbaik {float(row['abs_error_metode_terbaik_gram']):.3f} g | "
                f"alasan: {row['alasan_cek_ulang']}\n"
            )

    combined_graph, actual_combined_graph, metric_graph = save_summary_graphs(method_results, ranking_rows)
    pdf_path = save_pdf_report(ranking_rows, recheck_rows, text_lines, combined_graph, actual_combined_graph, metric_graph, method_results)
    docx_path = save_docx_report(ranking_rows, recheck_rows, text_lines, combined_graph, actual_combined_graph, metric_graph, method_results)
    ordered_folder = create_ordered_folder_copies(method_results, ranking_rows)

    return {
        "ranking_csv": ranking_csv,
        "worst_order_csv": worst_order_csv,
        "method_table_csv": method_table_csv,
        "recheck_csv": recheck_csv,
        "summary_txt": summary_txt,
        "combined_graph": combined_graph,
        "actual_combined_graph": actual_combined_graph,
        "metric_graph": metric_graph,
        "pdf_report": pdf_path,
        "docx_report": docx_path,
        "ordered_folder": ordered_folder,
    }


# =========================================================
# PDF / DOCX REPORT
# =========================================================
def add_text_page(pdf, title, lines, fontsize=10):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.05, 0.94, title, fontsize=16, weight="bold", va="top")
    y = 0.88
    for line in lines:
        chunks = wrap_text(line, 115)
        for chunk in chunks:
            fig.text(0.05, y, chunk, fontsize=fontsize, va="top")
            y -= 0.04
            if y < 0.08:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(11.69, 8.27))
                y = 0.92
        y -= 0.02
    pdf.savefig(fig)
    plt.close(fig)


def wrap_text(text, width):
    words = str(text).split()
    lines = []
    current = []
    for word in words:
        if sum(len(w) for w in current) + len(current) + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def save_pdf_report(ranking_rows, recheck_rows, text_lines, combined_graph, actual_combined_graph, metric_graph, method_results):
    if not MATPLOTLIB_OK:
        return ""
    path = os.path.join(SUMMARY_DIR, "00_laporan_perbandingan_5_model.pdf")
    with PdfPages(path) as pdf:
        if combined_graph and os.path.exists(combined_graph):
            img = plt.imread(combined_graph)
            fig = plt.figure(figsize=(11.69, 16.54))
            plt.imshow(img)
            plt.axis("off")
            plt.title("Gabungan 5 Grafik Volume ke Massa", fontsize=16)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        if actual_combined_graph and os.path.exists(actual_combined_graph):
            img = plt.imread(actual_combined_graph)
            fig = plt.figure(figsize=(11.69, 16.54))
            plt.imshow(img)
            plt.axis("off")
            plt.title("Gabungan 5 Grafik Aktual vs Prediksi", fontsize=16)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        if metric_graph and os.path.exists(metric_graph):
            img = plt.imread(metric_graph)
            fig = plt.figure(figsize=(11.69, 8.27))
            plt.imshow(img)
            plt.axis("off")
            plt.title("Perbandingan MAE, RMSE, dan MAPE", fontsize=16)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        method_lines = []
        for row in method_explanation_rows():
            method_lines.append(f"{row['method_label']}")
            method_lines.append(f"Rumus: {row['rumus']}")
            method_lines.append(f"Penjelasan: {row['penjelasan_metode']}")
            method_lines.append(f"Relevansi: {row['alasan_relevan']}")
            method_lines.append("")
        add_text_page(pdf, "Tabel Penjelasan 5 Model", method_lines, fontsize=8)

        rank_lines = [
            "Ranking memakai jumlah ranking MAE, RMSE, MAPE, R2, dan jumlah error tinggi.",
            "Semakin kecil ranking_score, semakin baik performa gabungan metode.",
            "Daftar berikut diurutkan dari error/ranking terburuk menuju metode terbaik.",
            "",
        ]
        rank_lines.extend(scoring_explanation_lines())
        rank_lines.append("")
        for row in worst_to_best_rows(ranking_rows):
            rank_lines.append(
                f"{row['method_label']} | ranking akhir {row['ranking_akhir']} | score {row['ranking_score']} | "
                f"MAE {float(row['mae_gram']):.3f} g | RMSE {float(row['rmse_gram']):.3f} g | "
                f"MAPE {float(row['mape_percent']):.3f}% | R2 {float(row['r2']):.6f} | "
                f"error tinggi {row['high_error_count']}"
            )
        add_text_page(pdf, "Tabel Metrik dan Urutan Metode", rank_lines, fontsize=8)

        add_text_page(pdf, "Analisis Metode Terbaik", text_lines, fontsize=10)

        recheck_lines = [
            "Data berikut bukan otomatis dibuang. Data ini diprioritaskan untuk dicek ulang karena error tinggi.",
            "Cek ulang: massa aktual, mask batubara, depth/height map, volume, baseline, dan bentuk muatan.",
            "",
        ]
        for row in recheck_rows[:30]:
            recheck_lines.append(
                f"{row['sample_id']} | baseline {row['baseline_cm']} cm | profil {row['profile']} | "
                f"aktual {float(row['actual_gram']):.1f} g | error terbaik "
                f"{float(row['abs_error_metode_terbaik_gram']):.1f} g | {row['alasan_cek_ulang']}"
            )
        add_text_page(pdf, "Data Prioritas Cek Ulang", recheck_lines, fontsize=8)
    return path


def save_docx_report(ranking_rows, recheck_rows, text_lines, combined_graph, actual_combined_graph, metric_graph, method_results):
    if not DOCX_OK:
        return ""
    path = os.path.join(SUMMARY_DIR, "00_laporan_perbandingan_5_model.docx")
    doc = Document()
    doc.add_heading("Laporan Perbandingan 5 Model Estimasi Massa Batubara", level=1)
    doc.add_paragraph(f"Waktu evaluasi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph(f"Sumber data: {CALIBRATION_CSV}")

    doc.add_heading("Grafik Gabungan Volume ke Massa", level=2)
    if combined_graph and os.path.exists(combined_graph):
        doc.add_picture(combined_graph, width=Inches(6.5))
    doc.add_heading("Grafik Gabungan Aktual vs Prediksi", level=2)
    if actual_combined_graph and os.path.exists(actual_combined_graph):
        doc.add_picture(actual_combined_graph, width=Inches(6.5))
    if metric_graph and os.path.exists(metric_graph):
        doc.add_picture(metric_graph, width=Inches(6.5))

    doc.add_heading("Penjelasan 5 Model", level=2)
    method_table = doc.add_table(rows=1, cols=4)
    method_headers = ["Metode", "Rumus", "Penjelasan", "Relevansi"]
    for i, h in enumerate(method_headers):
        method_table.rows[0].cells[i].text = h
    for row in method_explanation_rows():
        cells = method_table.add_row().cells
        cells[0].text = row["method_label"]
        cells[1].text = row["rumus"]
        cells[2].text = row["penjelasan_metode"]
        cells[3].text = row["alasan_relevan"]

    doc.add_heading("Ranking Metode dari Error Tertinggi ke Terbaik", level=2)
    for line in scoring_explanation_lines():
        doc.add_paragraph(line)

    table = doc.add_table(rows=1, cols=7)
    headers = ["Rank", "Metode", "MAE", "RMSE", "MAPE", "R2", "Error tinggi"]
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
    for row in worst_to_best_rows(ranking_rows):
        cells = table.add_row().cells
        cells[0].text = str(row["ranking_akhir"])
        cells[1].text = str(row["method_label"])
        cells[2].text = f"{float(row['mae_gram']):.3f}"
        cells[3].text = f"{float(row['rmse_gram']):.3f}"
        cells[4].text = f"{float(row['mape_percent']):.3f}"
        cells[5].text = f"{float(row['r2']):.6f}"
        cells[6].text = str(row["high_error_count"])

    doc.add_heading("Analisis", level=2)
    for line in text_lines:
        doc.add_paragraph(line)

    doc.add_heading("Data Prioritas Cek Ulang", level=2)
    for row in recheck_rows[:30]:
        doc.add_paragraph(
            f"{row['sample_id']} | baseline {row['baseline_cm']} cm | profil {row['profile']} | "
            f"error terbaik {float(row['abs_error_metode_terbaik_gram']):.1f} g | {row['alasan_cek_ulang']}",
            style=None,
        )

    doc.save(path)
    return path


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n" + "=" * 78)
    print("SCRIPT 18 - PERBANDINGAN 5 MODEL ESTIMASI MASSA BATUBARA")
    print("=" * 78)
    print(f"Data kalibrasi : {CALIBRATION_CSV}")
    print(f"Output root    : {OUTPUT_ROOT}")
    print("=" * 78)

    ensure_dir(OUTPUT_ROOT)
    ensure_dir(SUMMARY_DIR)

    data = load_calibration_data()
    if not data:
        print("\n[WARNING] Belum ada data valid.")
        print(f"Isi massa_aktual_gram di: {CALIBRATION_CSV}")
        return

    groups = group_by_baseline(data)
    print(f"\nJumlah data valid : {len(data)}")
    print("Baseline          : " + ", ".join(str(k) for k in sorted(groups.keys(), key=baseline_sort_key)))

    method_results = []
    for folder_name, method_label, method_key in METHODS:
        print(f"[RUN] {method_label}")
        result = save_method_outputs(folder_name, method_label, method_key, groups)
        method_results.append(result)

    ranking_rows = build_ranking(method_results)
    best_result = None
    if ranking_rows:
        best_folder = ranking_rows[0]["method_folder"]
        best_result = next((r for r in method_results if r["folder_name"] == best_folder), None)

    recheck_rows = build_recheck_rows(method_results, best_result)
    summary_paths = save_summary_files(method_results, ranking_rows, recheck_rows)
    train_test_paths = save_train_test_evaluation(groups)

    print("\nURUTAN METODE DARI ERROR TERTINGGI KE TERBAIK")
    print("---------------------------------------------")
    for row in worst_to_best_rows(ranking_rows):
        print(
            f"{row['method_label']} | ranking akhir {row['ranking_akhir']} | "
            f"MAE {float(row['mae_gram']):.3f} g | RMSE {float(row['rmse_gram']):.3f} g | "
            f"MAPE {float(row['mape_percent']):.3f}% | R2 {float(row['r2']):.5f} | "
            f"score {row['ranking_score']}"
        )

    print("\n[OK] Output tersimpan")
    print(f"Folder utama        : {OUTPUT_ROOT}")
    print(f"Ringkasan CSV       : {summary_paths['ranking_csv']}")
    print(f"Urutan worst-best   : {summary_paths['worst_order_csv']}")
    print(f"Tabel metode CSV    : {summary_paths['method_table_csv']}")
    print(f"Data cek ulang CSV  : {summary_paths['recheck_csv']}")
    print(f"Analisis TXT        : {summary_paths['summary_txt']}")
    print(f"Train-test metrik   : {train_test_paths['train_test_metrics_csv']}")
    print(f"Train-test prediksi : {train_test_paths['train_test_predictions_csv']}")
    print(f"Train-test catatan  : {train_test_paths['train_test_summary_txt']}")
    if summary_paths["combined_graph"]:
        print(f"Grafik volume-massa : {summary_paths['combined_graph']}")
    if summary_paths["actual_combined_graph"]:
        print(f"Grafik aktual-pred  : {summary_paths['actual_combined_graph']}")
    if summary_paths["pdf_report"]:
        print(f"Laporan PDF         : {summary_paths['pdf_report']}")
    if summary_paths["docx_report"]:
        print(f"Laporan DOCX        : {summary_paths['docx_report']}")
    else:
        print("Laporan DOCX        : tidak dibuat karena python-docx tidak tersedia")
    print(f"Folder urut         : {summary_paths['ordered_folder']}")


if __name__ == "__main__":
    main()
