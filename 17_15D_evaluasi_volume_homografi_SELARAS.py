"""
SCRIPT 15D — EVALUASI VOLUME HOMOGRAFI GERBONG BATUBARA (SELARAS SISTEM)
===========================================================================

Posisi dalam alur sistem:
    Script 11 -> Integrasi U-Net + depth + height map
    Script 12 -> Hitung volume dengan homografi ROI gerbong
    Script 15D -> Evaluasi hasil volume homografi
    Script 13B/14 -> Kalibrasi empiris massa dan evaluasi massa akhir

Fungsi utama:
1. Membaca hasil Script 12 dari:
       hasil_volume_homografi/volume_history.csv
2. Mengevaluasi kewajaran volume berdasarkan batas fisik gerbong:
       panjang = 40 cm, lebar = 10 cm, tinggi = 7.5 cm
       kapasitas maksimum = 3000 cm^3
3. Mengevaluasi tinggi maksimum agar tidak melebihi batas fisik gerbong
   dengan toleransi, misalnya 9 cm.
4. Mengevaluasi jumlah valid volume pixels agar hasil tidak berasal dari noise kecil.
5. Mengevaluasi area per pixel homografi jika data tersedia.
6. Opsional: membandingkan volume sistem dengan volume aktual/manual jika pengguna
   mengisi file:
       hasil_evaluasi_volume_homografi/data_volume_aktual_homografi.csv

Catatan:
- Script ini TIDAK menghitung ulang volume dan TIDAK menghitung ulang tinggi dari Z_ref global.
- Script ini hanya mengevaluasi hasil volume dari Script 12 terbaru yang memakai height_m dari Script 11.
- Jika volume aktual belum tersedia, script tetap membuat evaluasi kewajaran internal.
- Jika volume aktual tersedia, script menghitung MAE, RMSE, MAPE, R2, dan error per data.
"""

import os
import csv
import math
from glob import glob
from datetime import datetime

import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False


# =========================================================
# KONFIGURASI PATH
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOLUME_DIR = os.path.join(BASE_DIR, "hasil_volume_homografi")
VOLUME_HISTORY = os.path.join(VOLUME_DIR, "volume_history.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "hasil_evaluasi_volume_homografi")
AKTUAL_CSV = os.path.join(OUTPUT_DIR, "data_volume_aktual_homografi.csv")


# =========================================================
# KONFIGURASI DIMENSI FISIK GERBONG MINIATUR
# =========================================================
GERBONG_PANJANG_CM = 40.0
GERBONG_LEBAR_CM = 10.0
GERBONG_TINGGI_CM = 7.5

# Batas toleransi tinggi. Tinggi gerbong 7.5 cm, diberi margin karena permukaan batubara
# bisa sedikit menumpuk atau karena noise depth.
BATAS_MAX_HEIGHT_CM = 15.0

# Kapasitas fisik gerbong jika penuh rata setinggi dinding gerbong.
KAPASITAS_CM3 = GERBONG_PANJANG_CM * GERBONG_LEBAR_CM * GERBONG_TINGGI_CM

# Kapasitas maksimum yang masih diberi toleransi.
# Jika volume melebihi ini, hasil dianggap perlu dicek ulang.
TOLERANSI_KAPASITAS_PERSEN = 10.0
BATAS_VOLUME_MAKS_CM3 = KAPASITAS_CM3 * (1.0 + TOLERANSI_KAPASITAS_PERSEN / 100.0)

# Minimum piksel valid agar hasil tidak berasal dari noise kecil.
MIN_VALID_VOLUME_PIXELS = 50

# Parameter top-view Script 12: 10 px/cm -> 400 x 100 px.
TOPVIEW_PIXELS_PER_CM = 10
TOPVIEW_W = int(round(GERBONG_PANJANG_CM * TOPVIEW_PIXELS_PER_CM))
TOPVIEW_H = int(round(GERBONG_LEBAR_CM * TOPVIEW_PIXELS_PER_CM))
EXPECTED_AREA_PER_PIXEL_M2 = (GERBONG_PANJANG_CM / 100.0) * (GERBONG_LEBAR_CM / 100.0) / (TOPVIEW_W * TOPVIEW_H)
TOLERANSI_AREA_PER_PIXEL_PERSEN = 2.0


# =========================================================
# UTILITAS UMUM
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def norm_key(s):
    return str(s).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def parse_float(value, default=np.nan):
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return float(text.replace(",", "."))
    except Exception:
        return default


def get_first_float(row, candidates, default=np.nan):
    keys = {norm_key(k): k for k in row.keys()}
    for c in candidates:
        nk = norm_key(c)
        if nk in keys:
            val = parse_float(row.get(keys[nk]), default=np.nan)
            if np.isfinite(val):
                return val
    return default


def get_first_text(row, candidates, default=""):
    keys = {norm_key(k): k for k in row.keys()}
    for c in candidates:
        nk = norm_key(c)
        if nk in keys:
            val = str(row.get(keys[nk], "")).strip()
            if val:
                return val
    return default


def safe_div(a, b, default=np.nan):
    try:
        if b == 0 or not np.isfinite(b):
            return default
        return a / b
    except Exception:
        return default


# =========================================================
# MEMBACA HASIL SCRIPT 12
# =========================================================
def load_volume_history():
    if not os.path.exists(VOLUME_HISTORY):
        raise FileNotFoundError(
            "volume_history.csv tidak ditemukan. Jalankan Script 12 terlebih dahulu.\n"
            f"Path yang dicari: {VOLUME_HISTORY}"
        )

    with open(VOLUME_HISTORY, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if len(rows) == 0:
        raise RuntimeError("volume_history.csv kosong. Jalankan Script 12 dan pastikan hasil volume tersimpan.")

    data = []
    for i, row in enumerate(rows, start=1):
        prefix = get_first_text(row, ["prefix", "nama", "file", "snapshot", "snapshot_npz"], default=f"data_{i:03d}")

        volume_cm3 = get_first_float(row, ["volume_cm3", "volume_cm^3", "volume", "volume_batubara_cm3"])
        volume_liter = get_first_float(row, ["volume_liter", "volume_l", "liter"])
        volume_m3 = get_first_float(row, ["volume_m3", "volume_m^3"])

        if not np.isfinite(volume_cm3):
            if np.isfinite(volume_liter):
                volume_cm3 = volume_liter * 1000.0
            elif np.isfinite(volume_m3):
                volume_cm3 = volume_m3 * 1_000_000.0

        mean_h = get_first_float(row, ["mean_height_cm", "mean_cm", "mean_height"])
        median_h = get_first_float(row, ["median_height_cm", "median_cm", "median_height"])
        max_h = get_first_float(row, ["max_height_cm", "max_cm", "max_height"])
        valid_pixels = get_first_float(row, ["valid_pixels", "valid_volume_pixels", "valid_height_pixels"], default=0.0)
        area_per_pixel = get_first_float(row, ["area_per_pixel_m2", "area_pixel_m2", "luas_per_pixel_m2"])
        timestamp = get_first_text(row, ["timestamp", "waktu", "tanggal"], default="")

        # Jika area_per_pixel tidak ada di volume_history, coba baca dari NPZ output Script 12.
        if not np.isfinite(area_per_pixel):
            area_per_pixel = cari_area_per_pixel_dari_npz(prefix)

        data.append({
            "no": i,
            "timestamp": timestamp,
            "prefix": prefix,
            "volume_cm3": volume_cm3,
            "mean_height_cm": mean_h,
            "median_height_cm": median_h,
            "max_height_cm": max_h,
            "valid_pixels": valid_pixels,
            "area_per_pixel_m2": area_per_pixel,
        })

    return data


def cari_area_per_pixel_dari_npz(prefix):
    """Mencari area_per_pixel_m2 dari file *_volume_data.npz hasil Script 12."""
    candidates = sorted(glob(os.path.join(VOLUME_DIR, "*_volume_data.npz")), key=os.path.getmtime, reverse=True)
    # Usahakan cocok dengan prefix. Jika tidak ketemu, return NaN.
    for path in candidates:
        base = os.path.basename(path)
        if str(prefix) in base:
            try:
                npz = np.load(path)
                if "area_per_pixel_m2" in npz:
                    arr = np.asarray(npz["area_per_pixel_m2"], dtype=float).ravel()
                    if arr.size > 0 and np.isfinite(arr[0]):
                        return float(arr[0])
            except Exception:
                pass
    return np.nan


# =========================================================
# DATA AKTUAL OPSIONAL
# =========================================================
def buat_template_volume_aktual(data):
    """Membuat template CSV jika pengguna ingin membandingkan dengan volume aktual/manual."""
    if os.path.exists(AKTUAL_CSV):
        return False

    ensure_dir(OUTPUT_DIR)
    with open(AKTUAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prefix",
            "volume_aktual_cm3",
            "keterangan",
        ])
        for d in data:
            writer.writerow([
                d["prefix"],
                "",
                "isi volume aktual/manual jika ada; kosongkan jika belum ada",
            ])
    return True


def load_volume_aktual():
    """Membaca data volume aktual jika sudah diisi pengguna."""
    if not os.path.exists(AKTUAL_CSV):
        return {}

    aktual = {}
    with open(AKTUAL_CSV, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            prefix = get_first_text(row, ["prefix", "nama", "snapshot"], default="")
            vol = get_first_float(row, ["volume_aktual_cm3", "aktual_cm3", "volume_manual_cm3"])
            ket = get_first_text(row, ["keterangan", "catatan"], default="")
            if prefix and np.isfinite(vol):
                aktual[prefix] = {"volume_aktual_cm3": vol, "keterangan": ket}
    return aktual


# =========================================================
# EVALUASI
# =========================================================
def evaluasi_kewajaran(data, aktual_map):
    hasil = []

    lower_area = EXPECTED_AREA_PER_PIXEL_M2 * (1.0 - TOLERANSI_AREA_PER_PIXEL_PERSEN / 100.0)
    upper_area = EXPECTED_AREA_PER_PIXEL_M2 * (1.0 + TOLERANSI_AREA_PER_PIXEL_PERSEN / 100.0)

    for d in data:
        volume = d["volume_cm3"]
        max_h = d["max_height_cm"]
        valid_pixels = d["valid_pixels"]
        area_per_pixel = d["area_per_pixel_m2"]

        volume_lulus = np.isfinite(volume) and (0.0 <= volume <= BATAS_VOLUME_MAKS_CM3)
        max_height_lulus = np.isfinite(max_h) and (0.0 <= max_h <= BATAS_MAX_HEIGHT_CM)
        valid_pixels_lulus = np.isfinite(valid_pixels) and (valid_pixels >= MIN_VALID_VOLUME_PIXELS)

        if np.isfinite(area_per_pixel):
            area_pixel_lulus = lower_area <= area_per_pixel <= upper_area
            area_pixel_error_percent = abs(area_per_pixel - EXPECTED_AREA_PER_PIXEL_M2) / EXPECTED_AREA_PER_PIXEL_M2 * 100.0
        else:
            area_pixel_lulus = False
            area_pixel_error_percent = np.nan

        aktual = aktual_map.get(d["prefix"], None)
        volume_aktual = np.nan
        error_cm3 = np.nan
        abs_error_cm3 = np.nan
        error_persen = np.nan
        if aktual is not None:
            volume_aktual = aktual["volume_aktual_cm3"]
            error_cm3 = volume - volume_aktual if np.isfinite(volume) else np.nan
            abs_error_cm3 = abs(error_cm3) if np.isfinite(error_cm3) else np.nan
            error_persen = abs_error_cm3 / volume_aktual * 100.0 if volume_aktual != 0 else np.nan

        # Lulus internal berfokus pada kewajaran fisik dan kelengkapan data.
        lulus_internal = bool(volume_lulus and max_height_lulus and valid_pixels_lulus and area_pixel_lulus)

        out = dict(d)
        out.update({
            "kapasitas_fisik_cm3": KAPASITAS_CM3,
            "batas_volume_maks_cm3": BATAS_VOLUME_MAKS_CM3,
            "batas_max_height_cm": BATAS_MAX_HEIGHT_CM,
            "expected_area_per_pixel_m2": EXPECTED_AREA_PER_PIXEL_M2,
            "area_pixel_error_percent": area_pixel_error_percent,
            "volume_lulus": volume_lulus,
            "max_height_lulus": max_height_lulus,
            "valid_pixels_lulus": valid_pixels_lulus,
            "area_pixel_lulus": area_pixel_lulus,
            "lulus_internal": lulus_internal,
            "volume_aktual_cm3": volume_aktual,
            "error_cm3": error_cm3,
            "abs_error_cm3": abs_error_cm3,
            "error_percent": error_persen,
        })
        hasil.append(out)

    return hasil


def hitung_metrik_aktual(hasil):
    y_true = []
    y_pred = []
    for h in hasil:
        if np.isfinite(h["volume_aktual_cm3"]) and np.isfinite(h["volume_cm3"]):
            y_true.append(float(h["volume_aktual_cm3"]))
            y_pred.append(float(h["volume_cm3"]))

    if len(y_true) == 0:
        return None

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(abs_err / np.maximum(np.abs(y_true), 1e-9)) * 100.0)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "n": int(len(y_true)),
        "mae_cm3": mae,
        "rmse_cm3": rmse,
        "mape_percent": mape,
        "r2": r2,
    }


# =========================================================
# SIMPAN OUTPUT
# =========================================================
def simpan_csv(hasil, out_csv):
    fieldnames = [
        "no", "timestamp", "prefix",
        "volume_cm3", "volume_aktual_cm3", "error_cm3", "abs_error_cm3", "error_percent",
        "kapasitas_fisik_cm3", "batas_volume_maks_cm3",
        "mean_height_cm", "median_height_cm", "max_height_cm", "batas_max_height_cm",
        "valid_pixels", "area_per_pixel_m2", "expected_area_per_pixel_m2", "area_pixel_error_percent",
        "volume_lulus", "max_height_lulus", "valid_pixels_lulus", "area_pixel_lulus", "lulus_internal",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in hasil:
            writer.writerow(h)


def simpan_ringkasan_txt(hasil, metrik, out_txt):
    volumes = np.array([h["volume_cm3"] for h in hasil if np.isfinite(h["volume_cm3"])], dtype=float)
    heights = np.array([h["max_height_cm"] for h in hasil if np.isfinite(h["max_height_cm"])], dtype=float)
    valid_pixels = np.array([h["valid_pixels"] for h in hasil if np.isfinite(h["valid_pixels"])], dtype=float)

    lulus_count = sum(1 for h in hasil if h["lulus_internal"])
    gagal = [h for h in hasil if not h["lulus_internal"]]

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("EVALUASI VOLUME HOMOGRAFI GERBONG BATUBARA\n")
        f.write("================================================\n\n")
        f.write("Posisi evaluasi: setelah Script 12 dan sebelum evaluasi massa akhir.\n")
        f.write(f"Input volume history : {VOLUME_HISTORY}\n")
        f.write(f"Data aktual opsional : {AKTUAL_CSV}\n")
        f.write(f"Jumlah data          : {len(hasil)}\n\n")

        f.write("Dasar fisik gerbong:\n")
        f.write(f"- Panjang gerbong     : {GERBONG_PANJANG_CM:.3f} cm\n")
        f.write(f"- Lebar gerbong       : {GERBONG_LEBAR_CM:.3f} cm\n")
        f.write(f"- Tinggi gerbong      : {GERBONG_TINGGI_CM:.3f} cm\n")
        f.write(f"- Kapasitas fisik     : {KAPASITAS_CM3:.3f} cm3\n")
        f.write(f"- Batas volume maks   : {BATAS_VOLUME_MAKS_CM3:.3f} cm3\n")
        f.write(f"- Batas max height    : {BATAS_MAX_HEIGHT_CM:.3f} cm\n")
        f.write(f"- Min valid pixels    : {MIN_VALID_VOLUME_PIXELS}\n")
        f.write(f"- Area/pixel harapan  : {EXPECTED_AREA_PER_PIXEL_M2:.12f} m2\n\n")

        f.write("Hasil evaluasi internal:\n")
        f.write(f"- Data lulus          : {lulus_count}/{len(hasil)}\n")
        f.write(f"- Data perlu dicek    : {len(gagal)}\n\n")

        if volumes.size > 0:
            f.write("Statistik volume sistem:\n")
            f.write(f"- Minimum             : {np.min(volumes):.6f} cm3\n")
            f.write(f"- Rata-rata           : {np.mean(volumes):.6f} cm3\n")
            f.write(f"- Median              : {np.median(volumes):.6f} cm3\n")
            f.write(f"- Maksimum            : {np.max(volumes):.6f} cm3\n")
            if volumes.size > 1 and abs(np.mean(volumes)) > 1e-9:
                cv = np.std(volumes, ddof=1) / np.mean(volumes) * 100.0
                f.write(f"- CV                  : {cv:.6f}%\n")
            f.write("\n")

        if heights.size > 0:
            f.write("Statistik max height:\n")
            f.write(f"- Minimum             : {np.min(heights):.6f} cm\n")
            f.write(f"- Rata-rata           : {np.mean(heights):.6f} cm\n")
            f.write(f"- Maksimum            : {np.max(heights):.6f} cm\n\n")

        if valid_pixels.size > 0:
            f.write("Statistik valid volume pixels:\n")
            f.write(f"- Minimum             : {np.min(valid_pixels):.0f}\n")
            f.write(f"- Rata-rata           : {np.mean(valid_pixels):.2f}\n")
            f.write(f"- Maksimum            : {np.max(valid_pixels):.0f}\n\n")

        if metrik is not None:
            f.write("Evaluasi terhadap volume aktual/manual:\n")
            f.write(f"- Jumlah data aktual  : {metrik['n']}\n")
            f.write(f"- MAE                 : {metrik['mae_cm3']:.6f} cm3\n")
            f.write(f"- RMSE                : {metrik['rmse_cm3']:.6f} cm3\n")
            f.write(f"- MAPE                : {metrik['mape_percent']:.6f}%\n")
            f.write(f"- R2                  : {metrik['r2']:.6f}\n\n")
        else:
            f.write("Evaluasi terhadap volume aktual/manual belum dilakukan karena data aktual belum diisi.\n")
            f.write("Isi file data_volume_aktual_homografi.csv jika ingin membandingkan volume sistem dengan volume aktual.\n\n")

        if gagal:
            f.write("Data yang perlu dicek ulang:\n")
            for h in gagal:
                alasan = []
                if not h["volume_lulus"]:
                    alasan.append("volume di luar batas fisik")
                if not h["max_height_lulus"]:
                    alasan.append("max height melebihi batas")
                if not h["valid_pixels_lulus"]:
                    alasan.append("valid volume pixels terlalu sedikit")
                if not h["area_pixel_lulus"]:
                    alasan.append("area per pixel tidak sesuai/ tidak terbaca")
                f.write(f"- {h['prefix']}: {', '.join(alasan)}\n")
            f.write("\n")

        f.write("Interpretasi:\n")
        f.write("Evaluasi internal memastikan hasil volume masih masuk akal terhadap dimensi fisik gerbong, ")
        f.write("jumlah piksel valid tersedia, tinggi maksimum tidak berlebihan, dan area per piksel sesuai dengan hasil homografi. ")
        f.write("Jika data aktual tersedia, evaluasi absolut dilakukan dengan membandingkan volume sistem terhadap volume aktual/manual.\n")


def simpan_grafik(hasil, metrik, out_dir):
    if not MATPLOTLIB_OK:
        print("[WARNING] matplotlib tidak tersedia, grafik tidak dibuat.")
        return

    no = [h["no"] for h in hasil]
    volume = [h["volume_cm3"] for h in hasil]
    max_height = [h["max_height_cm"] for h in hasil]
    valid_pixels = [h["valid_pixels"] for h in hasil]

    plt.figure(figsize=(8, 4.5))
    plt.plot(no, volume, marker="o", label="Volume sistem")
    plt.axhline(KAPASITAS_CM3, linestyle="--", label="Kapasitas fisik gerbong")
    plt.axhline(BATAS_VOLUME_MAKS_CM3, linestyle=":", label="Batas toleransi volume")
    plt.xlabel("Data ke-")
    plt.ylabel("Volume (cm3)")
    plt.title("Evaluasi Kewajaran Volume Homografi")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "grafik_volume_homografi.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(no, max_height, marker="o", label="Max height")
    plt.axhline(GERBONG_TINGGI_CM, linestyle="--", label="Tinggi fisik gerbong")
    plt.axhline(BATAS_MAX_HEIGHT_CM, linestyle=":", label="Batas max height")
    plt.xlabel("Data ke-")
    plt.ylabel("Max height (cm)")
    plt.title("Evaluasi Max Height Volume Homografi")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "grafik_max_height_homografi.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(no, valid_pixels)
    plt.axhline(MIN_VALID_VOLUME_PIXELS, linestyle="--", label="Minimum valid pixels")
    plt.xlabel("Data ke-")
    plt.ylabel("Valid volume pixels")
    plt.title("Valid Volume Pixels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "grafik_valid_volume_pixels.png"), dpi=160)
    plt.close()

    # Grafik aktual vs sistem jika data aktual ada.
    actual_pairs = [(h["volume_aktual_cm3"], h["volume_cm3"]) for h in hasil if np.isfinite(h["volume_aktual_cm3"]) and np.isfinite(h["volume_cm3"])]
    if actual_pairs:
        y_true = np.array([p[0] for p in actual_pairs], dtype=float)
        y_pred = np.array([p[1] for p in actual_pairs], dtype=float)
        mn = min(float(np.min(y_true)), float(np.min(y_pred)))
        mx = max(float(np.max(y_true)), float(np.max(y_pred)))
        pad = max((mx - mn) * 0.05, 1.0)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred)
        plt.plot([mn - pad, mx + pad], [mn - pad, mx + pad], linestyle="--", label="Ideal")
        plt.xlabel("Volume aktual/manual (cm3)")
        plt.ylabel("Volume sistem (cm3)")
        title = "Volume Aktual vs Volume Sistem"
        if metrik is not None:
            title += f"\nMAE={metrik['mae_cm3']:.2f} cm3 | MAPE={metrik['mape_percent']:.2f}%"
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "grafik_volume_aktual_vs_sistem.png"), dpi=160)
        plt.close()


def main():
    print("\n" + "=" * 78)
    print("SCRIPT 15D — EVALUASI VOLUME HOMOGRAFI GERBONG BATUBARA")
    print("=" * 78)
    print(f"Input Script 12 : {VOLUME_HISTORY}")
    print(f"Output          : {OUTPUT_DIR}")
    print(f"Kapasitas       : {KAPASITAS_CM3:.2f} cm3")
    print(f"Batas volume    : {BATAS_VOLUME_MAKS_CM3:.2f} cm3")
    print(f"Batas height    : {BATAS_MAX_HEIGHT_CM:.2f} cm")
    print("=" * 78)

    ensure_dir(OUTPUT_DIR)

    data = load_volume_history()
    created = buat_template_volume_aktual(data)
    if created:
        print(f"[OK] Template volume aktual dibuat: {AKTUAL_CSV}")
        print("     Jika belum punya volume aktual, template boleh dibiarkan kosong.")

    aktual_map = load_volume_aktual()
    if aktual_map:
        print(f"[OK] Data volume aktual terbaca: {len(aktual_map)} data")
    else:
        print("[INFO] Data volume aktual belum diisi. Evaluasi yang dilakukan adalah evaluasi kewajaran internal.")

    hasil = evaluasi_kewajaran(data, aktual_map)
    metrik = hitung_metrik_aktual(hasil)

    waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(OUTPUT_DIR, f"evaluasi_volume_homografi_SELARAS_{waktu}.csv")
    out_txt = os.path.join(OUTPUT_DIR, f"ringkasan_evaluasi_volume_homografi_SELARAS_{waktu}.txt")

    simpan_csv(hasil, out_csv)
    simpan_ringkasan_txt(hasil, metrik, out_txt)
    simpan_grafik(hasil, metrik, OUTPUT_DIR)

    lulus_count = sum(1 for h in hasil if h["lulus_internal"])

    print("\n" + "=" * 78)
    print("HASIL EVALUASI VOLUME HOMOGRAFI")
    print("=" * 78)
    print(f"Jumlah data diproses : {len(hasil)}")
    print(f"Data lulus internal  : {lulus_count}/{len(hasil)}")
    if metrik is not None:
        print(f"MAE volume           : {metrik['mae_cm3']:.3f} cm3")
        print(f"RMSE volume          : {metrik['rmse_cm3']:.3f} cm3")
        print(f"MAPE volume          : {metrik['mape_percent']:.3f}%")
        print(f"R2 volume            : {metrik['r2']:.4f}")
    else:
        print("Metrik aktual        : belum dihitung karena volume aktual belum diisi")
    print(f"CSV hasil            : {out_csv}")
    print(f"TXT ringkasan        : {out_txt}")
    print(f"Template aktual      : {AKTUAL_CSV}")
    print(f"Folder output        : {OUTPUT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
