import csv
import glob
import json
import os
from datetime import datetime

import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False


# =========================================================
# SCRIPT 19 - ESTIMASI MASSA SEMUA MODEL
# =========================================================
# Input  : output Script 12 berupa volume_history.csv atau batch_volume_summary_*.csv
# Model  : semua *_model_dan_metrik.json dari Script 18
# Output : prediksi massa dari semua model agar bisa dibandingkan/dipilih
# =========================================================


# =========================================================
# KONFIGURASI
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPT12_OUTPUT_DIR = os.path.join(BASE_DIR, "hasil_volume_homografi")
SCRIPT18_MODEL_ROOT = os.path.join(BASE_DIR, "hasil_18_perbandingan_5_model_estimasi_massa")
OUTPUT_DIR = os.path.join(BASE_DIR, "hasil_19_estimasi_massa_semua_model")

# Kosongkan agar otomatis mengambil CSV terbaru dari Script 12.
# Bisa diisi manual, contoh:
# INPUT_VOLUME_CSV = r"D:\folder\hasil_volume_homografi\batch_volume_summary_20260704_084751.csv"
INPUT_VOLUME_CSV = ""

# "all" = tampilkan semua prediksi.
# Jika ingin rekomendasi utama tertentu, isi salah satu:
# density_manual, density_empirical, linear, piecewise, multiple_profile, piecewise_profile
MODEL_DIPAKAI = "all"

FALLBACK_AREA_PER_PIXEL_CM2 = 0.01
SAVE_VISUAL_DASHBOARD = True
SAVE_MODEL_GRAPH = True
MAX_VISUAL_DASHBOARDS = 100

METHOD_ORDER = [
    ("density_manual", "Densitas Manual"),
    ("density_empirical", "Densitas Empiris"),
    ("linear", "Linear Regression"),
    ("piecewise", "Piecewise Linear"),
    ("multiple_profile", "Multiple Regression + Profile"),
    ("piecewise_profile", "Piecewise Linear + Profile"),
]

REKOMENDASI_PRIORITAS = [
    "piecewise_profile",
    "multiple_profile",
    "piecewise",
    "linear",
    "density_empirical",
    "density_manual",
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


# =========================================================
# UTILITAS
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_float(value, default=None):
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return default


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_profile(value):
    text = str(value or "").strip().lower()
    if text == "":
        return "unknown"
    text = text.replace(" ", "_")
    return "".join(ch for ch in text if ch.isalnum() or ch == "_") or "unknown"


def get_first_float(row, keys, default=None):
    for key in keys:
        value = parse_float(row.get(key), None)
        if value is not None:
            return value
    return default


def get_volume_cm3(row):
    return get_first_float(row, ["volume_cm3", "volume_est_cm3"], None)


def get_area_batubara_cm2(row, valid_pixels):
    area_direct = get_first_float(row, ["area_batubara_cm2", "coal_area_cm2", "luas_batubara_cm2"], None)
    if area_direct is not None:
        return area_direct

    if valid_pixels is None:
        return None

    area_per_pixel_m2 = get_first_float(row, ["area_per_pixel_m2"], None)
    if area_per_pixel_m2 is not None:
        return float(valid_pixels) * float(area_per_pixel_m2) * 10000.0

    area_per_pixel_cm2 = get_first_float(row, ["area_per_pixel_cm2"], FALLBACK_AREA_PER_PIXEL_CM2)
    return float(valid_pixels) * float(area_per_pixel_cm2)


def normalize_baseline_key(value):
    number = parse_float(value, None)
    if number is None:
        return str(value or "").strip()
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return str(number).rstrip("0").rstrip(".")


def find_latest_volume_csv():
    if INPUT_VOLUME_CSV:
        return INPUT_VOLUME_CSV

    candidates = []
    candidates.extend(glob.glob(os.path.join(SCRIPT12_OUTPUT_DIR, "batch_volume_summary_*.csv")))
    candidates.extend(glob.glob(os.path.join(SCRIPT12_OUTPUT_DIR, "volume_history.csv")))
    candidates.extend(glob.glob(os.path.join(BASE_DIR, "batch_volume_summary_*.csv")))
    candidates.extend(glob.glob(os.path.join(BASE_DIR, "volume_history.csv")))

    candidates = [path for path in candidates if os.path.isfile(path)]
    if not candidates:
        return ""
    return max(candidates, key=os.path.getmtime)


def normalize_volume_row(row, no):
    valid_pixels = get_first_float(row, ["valid_pixels", "valid_height_px"], None)
    return {
        "no": no,
        "timestamp": row.get("timestamp", ""),
        "sample_id": row.get("sample_id", row.get("prefix", f"sample_{no:03d}")),
        "prefix": row.get("prefix", ""),
        "snapshot_npz": row.get("snapshot_npz", row.get("npz_path", "")),
        "volume_npz": row.get("volume_npz", ""),
        "baseline_cm": row.get("baseline_cm", ""),
        "target_weight_g": row.get("target_weight_g", row.get("target_massa_gram", "")),
        "actual_weight_g": row.get("actual_weight_g", row.get("massa_aktual_gram", "")),
        "condition": row.get("condition", ""),
        "profile": row.get("profile", ""),
        "profile_norm": normalize_profile(row.get("profile", "")),
        "repetition": row.get("repetition", ""),
        "volume_cm3": get_volume_cm3(row),
        "volume_liter": get_first_float(row, ["volume_liter"], None),
        "volume_m3": get_first_float(row, ["volume_m3"], None),
        "mean_height_cm": get_first_float(row, ["mean_height_cm"], None),
        "median_height_cm": get_first_float(row, ["median_height_cm"], None),
        "max_height_cm": get_first_float(row, ["max_height_cm"], None),
        "valid_pixels": valid_pixels,
        "area_batubara_cm2": get_area_batubara_cm2(row, valid_pixels),
        "p95_height_cm": get_first_float(row, ["p95_height_cm"], None),
        "p98_height_cm": get_first_float(row, ["p98_height_cm"], None),
        "status_script12": row.get("status", ""),
    }


def load_volume_rows(path):
    rows_raw = read_csv(path)
    rows = []
    for idx, row in enumerate(rows_raw, start=1):
        item = normalize_volume_row(row, idx)
        if item["volume_cm3"] is not None:
            rows.append(item)
    return rows


def find_volume_npz(row):
    candidates = []
    if row.get("volume_npz"):
        candidates.append(row["volume_npz"])

    prefix = row.get("prefix") or row.get("sample_id")
    if prefix:
        candidates.append(os.path.join(SCRIPT12_OUTPUT_DIR, f"{prefix}_volume_data.npz"))
        candidates.extend(glob.glob(os.path.join(SCRIPT12_OUTPUT_DIR, f"*{prefix}*_volume_data.npz")))

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return ""


# =========================================================
# LOAD MODEL SCRIPT 18
# =========================================================
def load_model_files():
    patterns = [
        os.path.join(SCRIPT18_MODEL_ROOT, "*", "*_model_dan_metrik.json"),
        os.path.join(BASE_DIR, "hasil_18_perbandingan_5_model_estimasi_massa", "*", "*_model_dan_metrik.json"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    unique_files = sorted(set(files))
    models = {}
    for path in unique_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        method_key = data.get("method_key", "")
        if method_key:
            data["_model_json_path"] = path
            models[method_key] = data
    return models


def choose_model_for_baseline(model_data, baseline_cm):
    models_by_baseline = model_data.get("models_by_baseline_cm", {})
    if not models_by_baseline:
        return "", None

    baseline_key = normalize_baseline_key(baseline_cm)
    if baseline_key in models_by_baseline:
        model_record = models_by_baseline[baseline_key]
        return baseline_key, model_record.get("model", model_record)

    numeric_target = parse_float(baseline_cm, None)
    numeric_candidates = []
    for key in models_by_baseline.keys():
        numeric_key = parse_float(key, None)
        if numeric_key is not None:
            numeric_candidates.append((abs(numeric_key - numeric_target), key))

    if numeric_target is not None and numeric_candidates:
        _, nearest_key = sorted(numeric_candidates)[0]
        model_record = models_by_baseline[nearest_key]
        return nearest_key, model_record.get("model", model_record)

    first_key = sorted(models_by_baseline.keys())[0]
    model_record = models_by_baseline[first_key]
    return first_key, model_record.get("model", model_record)


# =========================================================
# PREDIKSI MODEL
# =========================================================
def numeric_matrix(rows, features, existing_stats=None):
    cols = []
    stats = {}
    for feature in features:
        vals = np.array(
            [np.nan if row.get(feature) is None else float(row.get(feature)) for row in rows],
            dtype=float,
        )
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


def segment_index(value, breaks):
    idx = 0
    for br in breaks:
        if value > br:
            idx += 1
    return idx


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
            profile = row.get("profile_norm", "unknown")
            offsets = seg.get("profile_offsets", {})
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


def predict_method(method_key, rows, model_info):
    volume = np.array([row["volume_cm3"] for row in rows], dtype=float)

    if method_key == "density_manual":
        density = float(model_info.get("density_g_per_cm3", 0.0))
        return np.maximum(density * volume, 0.0)

    if method_key == "density_empirical":
        density = float(model_info.get("density_g_per_cm3", 0.0))
        return np.maximum(density * volume, 0.0)

    if method_key == "linear":
        return np.maximum(float(model_info["a"]) * volume + float(model_info["b"]), 0.0)

    if method_key == "piecewise":
        return predict_piecewise_from_model(rows, model_info, include_profile=False)

    if method_key == "piecewise_profile":
        return predict_piecewise_from_model(rows, model_info, include_profile=True)

    if method_key == "multiple_profile":
        return predict_multiple_from_model(rows, model_info)

    raise ValueError(f"Metode belum didukung Script 19: {method_key}")


def pilih_model_rekomendasi(row_result, available_method_keys):
    if MODEL_DIPAKAI != "all" and MODEL_DIPAKAI in available_method_keys:
        return MODEL_DIPAKAI

    for method_key in REKOMENDASI_PRIORITAS:
        if method_key in available_method_keys and row_result.get(f"pred_{method_key}_g") not in ["", None]:
            return method_key

    return available_method_keys[0] if available_method_keys else ""


# =========================================================
# VISUALISASI
# =========================================================
def safe_float_from_output(row, key, default=0.0):
    return parse_float(row.get(key), default)


def bgr_to_rgb_if_needed(img):
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3][:, :, ::-1]
    return arr


def height_to_cm(height_m):
    arr = np.asarray(height_m, dtype=float)
    return np.maximum(arr * 100.0, 0.0)


def save_sample_dashboard(volume_row, output_row, dashboard_dir):
    if not MATPLOTLIB_OK:
        return ""

    volume_npz = find_volume_npz(volume_row)
    if not volume_npz:
        return ""

    try:
        data = np.load(volume_npz)
    except Exception:
        return ""

    rect_top = data["rect_top"] if "rect_top" in data else None
    mask_top = data["mask_top"].astype(bool) if "mask_top" in data else None
    valid_volume = data["valid_volume"].astype(bool) if "valid_volume" in data else mask_top
    height_top_m = data["height_top_m"] if "height_top_m" in data else None

    if rect_top is None or mask_top is None or height_top_m is None:
        return ""

    height_cm = height_to_cm(height_top_m)
    overlay = bgr_to_rgb_if_needed(rect_top).copy()
    if overlay.ndim == 2:
        overlay = np.dstack([overlay, overlay, overlay])
    overlay = overlay.astype(float)
    red = np.zeros_like(overlay)
    red[..., 0] = 255.0
    if valid_volume is not None:
        overlay[valid_volume] = 0.55 * overlay[valid_volume] + 0.45 * red[valid_volume]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    sample_id = str(output_row.get("sample_id", f"sample_{output_row.get('no', '')}")).replace("\\", "_").replace("/", "_")
    path = os.path.join(dashboard_dir, f"{sample_id}_dashboard_estimasi.png")

    volume = safe_float_from_output(output_row, "volume_cm3")
    massa_g = safe_float_from_output(output_row, "massa_rekomendasi_g")
    model_name = output_row.get("model_rekomendasi", "")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Dashboard Estimasi Massa - {sample_id}", fontsize=12)

    axes[0].imshow(bgr_to_rgb_if_needed(rect_top))
    axes[0].set_title("1. Citra Top-View")
    axes[0].axis("off")

    axes[1].imshow(mask_top, cmap="gray")
    axes[1].set_title("2. Masker Area")
    axes[1].axis("off")

    im = axes[2].imshow(height_cm, cmap="turbo")
    axes[2].set_title("3. Height Map")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="cm")

    axes[3].imshow(overlay)
    axes[3].set_title("4. Estimasi Akhir")
    axes[3].axis("off")
    info = (
        f"Vol: {volume:.2f} cm3\n"
        f"Massa: {massa_g:.3f} g\n"
        f"Model: {model_name}"
    )
    axes[3].text(
        0.02, 0.98, info,
        transform=axes[3].transAxes,
        va="top", ha="left",
        fontsize=9, color="white",
        bbox=dict(facecolor="darkred", alpha=0.85, edgecolor="none"),
    )

    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_model_comparison_graph(output_rows, graph_dir):
    if not MATPLOTLIB_OK or not output_rows:
        return ""

    method_keys = [method_key for method_key, _label in METHOD_ORDER]
    method_labels = dict(METHOD_ORDER)

    path = os.path.join(graph_dir, "19_grafik_perbandingan_prediksi_semua_model.png")
    sample_labels = [str(row.get("sample_id", row.get("no", ""))) for row in output_rows]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(output_rows))
    plotted = False

    for method_key in method_keys:
        values = [parse_float(row.get(f"pred_{method_key}_g"), np.nan) for row in output_rows]
        if np.all(~np.isfinite(values)):
            continue
        ax.plot(x, values, marker="o", linewidth=1.6, label=method_labels.get(method_key, method_key))
        plotted = True

    actual_values = [parse_float(row.get("actual_weight_g"), np.nan) for row in output_rows]
    if np.any(np.isfinite(actual_values)):
        ax.plot(x, actual_values, linestyle="--", color="black", linewidth=2.0, label="Aktual/Timbangan")

    ax.set_title("Perbandingan Prediksi Massa Semua Model")
    ax.set_xlabel("Sampel")
    ax.set_ylabel("Massa Prediksi (gram)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    if len(sample_labels) <= 25:
        ax.set_xticks(x)
        ax.set_xticklabels(sample_labels, rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xticks(x[::max(1, len(x) // 20)])

    plt.tight_layout()
    if plotted:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path if plotted else ""


def save_visual_outputs(volume_rows, output_rows, timestamp):
    visual_paths = {
        "dashboard_dir": "",
        "comparison_graph": "",
        "dashboard_count": 0,
    }
    if not MATPLOTLIB_OK:
        return visual_paths

    dashboard_dir = os.path.join(OUTPUT_DIR, f"dashboard_visual_{timestamp}")
    graph_dir = os.path.join(OUTPUT_DIR, f"grafik_model_{timestamp}")
    ensure_dir(dashboard_dir)
    ensure_dir(graph_dir)

    if SAVE_VISUAL_DASHBOARD:
        for volume_row, output_row in zip(volume_rows[:MAX_VISUAL_DASHBOARDS], output_rows[:MAX_VISUAL_DASHBOARDS]):
            path = save_sample_dashboard(volume_row, output_row, dashboard_dir)
            if path:
                visual_paths["dashboard_count"] += 1
        visual_paths["dashboard_dir"] = dashboard_dir

    if SAVE_MODEL_GRAPH:
        visual_paths["comparison_graph"] = save_model_comparison_graph(output_rows, graph_dir)

    return visual_paths


# =========================================================
# PROSES UTAMA
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)

    input_csv = find_latest_volume_csv()
    if not input_csv:
        print("[ERROR] CSV hasil Script 12 tidak ditemukan.")
        print(f"Dicari di: {SCRIPT12_OUTPUT_DIR}")
        return

    volume_rows = load_volume_rows(input_csv)
    if not volume_rows:
        print("[ERROR] Tidak ada baris valid dengan volume_cm3 pada CSV input.")
        print(f"Input: {input_csv}")
        return

    model_datas = load_model_files()
    if not model_datas:
        print("[ERROR] Model dari Script 18 belum ditemukan.")
        print(f"Jalankan Script 18 dulu, lalu pastikan folder ini ada: {SCRIPT18_MODEL_ROOT}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(OUTPUT_DIR, f"19_estimasi_massa_semua_model_{timestamp}.csv")
    output_txt = os.path.join(OUTPUT_DIR, f"19_ringkasan_estimasi_massa_{timestamp}.txt")

    available_method_keys = [method_key for method_key, _ in METHOD_ORDER if method_key in model_datas]
    output_rows = []

    for row in volume_rows:
        out = {
            "no": row["no"],
            "sample_id": row["sample_id"],
            "prefix": row["prefix"],
            "baseline_cm_input": row["baseline_cm"],
            "target_weight_g": row["target_weight_g"],
            "actual_weight_g": row["actual_weight_g"],
            "condition": row["condition"],
            "profile": row["profile"],
            "profile_norm": row["profile_norm"],
            "volume_cm3": csv_value(row["volume_cm3"]),
            "volume_liter": csv_value(row["volume_liter"]),
            "mean_height_cm": csv_value(row["mean_height_cm"]),
            "median_height_cm": csv_value(row["median_height_cm"]),
            "max_height_cm": csv_value(row["max_height_cm"]),
            "valid_pixels": csv_value(row["valid_pixels"]),
            "area_batubara_cm2": csv_value(row["area_batubara_cm2"]),
            "p95_height_cm": csv_value(row["p95_height_cm"]),
            "p98_height_cm": csv_value(row["p98_height_cm"]),
            "status_script12": row["status_script12"],
        }

        for method_key, _method_label in METHOD_ORDER:
            pred_col = f"pred_{method_key}_g"
            baseline_col = f"baseline_model_{method_key}_cm"
            out[pred_col] = ""
            out[baseline_col] = ""

            model_data = model_datas.get(method_key)
            if not model_data:
                continue

            baseline_key, model_info = choose_model_for_baseline(model_data, row["baseline_cm"])
            if not model_info:
                continue

            pred = predict_method(method_key, [row], model_info)
            out[pred_col] = csv_value(float(pred[0]))
            out[baseline_col] = baseline_key

        rekomendasi_key = pilih_model_rekomendasi(out, available_method_keys)
        out["model_rekomendasi"] = rekomendasi_key
        out["massa_rekomendasi_g"] = out.get(f"pred_{rekomendasi_key}_g", "") if rekomendasi_key else ""

        output_rows.append(out)

    fieldnames = [
        "no", "sample_id", "prefix", "baseline_cm_input",
        "target_weight_g", "actual_weight_g", "condition", "profile", "profile_norm",
        "volume_cm3", "volume_liter",
        "mean_height_cm", "median_height_cm", "max_height_cm",
        "valid_pixels", "area_batubara_cm2", "p95_height_cm", "p98_height_cm",
        "status_script12",
    ]
    for method_key, _method_label in METHOD_ORDER:
        fieldnames.append(f"pred_{method_key}_g")
        fieldnames.append(f"baseline_model_{method_key}_cm")
    fieldnames.extend(["model_rekomendasi", "massa_rekomendasi_g"])

    write_csv(output_csv, output_rows, fieldnames)
    visual_paths = save_visual_outputs(volume_rows, output_rows, timestamp)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("RINGKASAN SCRIPT 19 - ESTIMASI MASSA SEMUA MODEL\n")
        f.write("=" * 64 + "\n")
        f.write(f"Waktu proses       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input Script 12    : {input_csv}\n")
        f.write(f"Folder model S18   : {SCRIPT18_MODEL_ROOT}\n")
        f.write(f"Jumlah data        : {len(output_rows)}\n")
        f.write(f"Model tersedia     : {', '.join(available_method_keys)}\n")
        f.write(f"Mode rekomendasi   : {MODEL_DIPAKAI}\n")
        f.write(f"Output CSV         : {output_csv}\n\n")
        f.write(f"Dashboard visual   : {visual_paths.get('dashboard_dir', '')}\n")
        f.write(f"Jumlah dashboard   : {visual_paths.get('dashboard_count', 0)}\n")
        f.write(f"Grafik model       : {visual_paths.get('comparison_graph', '')}\n\n")
        f.write("Catatan:\n")
        f.write("- Script ini tidak melatih ulang model.\n")
        f.write("- Script ini hanya membaca model Script 18 dan menerapkan ke data baru Script 12.\n")
        f.write("- Jika baseline input tidak ada di model, baseline model terdekat dipakai dan dicatat di kolom baseline_model_*_cm.\n")
        f.write("- Kolom massa_rekomendasi_g mengikuti MODEL_DIPAKAI. Jika MODEL_DIPAKAI = all, prioritas default adalah Piecewise Linear + Profile.\n")
        f.write("- Dashboard visual dibuat jika matplotlib tersedia dan file volume_npz dari Script 12 ditemukan.\n")

    print("\nSELESAI - SCRIPT 19 ESTIMASI MASSA SEMUA MODEL")
    print("=" * 64)
    print(f"Input Script 12 : {input_csv}")
    print(f"Jumlah data     : {len(output_rows)}")
    print(f"Model tersedia  : {', '.join(available_method_keys)}")
    print(f"Output CSV      : {output_csv}")
    print(f"Ringkasan TXT   : {output_txt}")
    print(f"Dashboard visual: {visual_paths.get('dashboard_dir', '')} ({visual_paths.get('dashboard_count', 0)} file)")
    print(f"Grafik model    : {visual_paths.get('comparison_graph', '')}")


if __name__ == "__main__":
    main()
