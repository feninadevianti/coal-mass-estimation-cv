import os
import sys
import csv
import json
import time
import importlib.util
from datetime import datetime
from collections import deque

import cv2
import numpy as np


# =========================================================
# KONFIGURASI UMUM
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPT11_PATH = os.path.join(BASE_DIR, "15_11_integrasi_unet_depth_heightmap_METADATA_KALIBRASI_MASSA.py")
SCRIPT12_PATH = os.path.join(BASE_DIR, "16_12_hitung_volume_homografi_BATCH_METADATA_KALIBRASI_MASSA.py")

OUTPUT_DIR = os.path.join(BASE_DIR, "hasil_demo_realtime")
OUTPUT_HISTORY_CSV = os.path.join(OUTPUT_DIR, "demo_realtime_history.csv")

WEIGHT_MODEL_JSON = os.path.join(BASE_DIR, "hasil_kalibrasi_massa", "model_kalibrasi_massa_per_baseline.json")

# Pilih baseline kamera yang sedang dipakai saat demo.
# Harus sesuai dengan stereomap.xml, zref, dan model kalibrasi massa yang digunakan.
DEMO_BASELINE_CM = 10.0  # ubah ke 15.0 jika memakai konfigurasi baseline 15 cm

# Jika model empiris massa dari Script 13B belum ada, sistem dapat memakai densitas sementara.
# Untuk laporan/validasi akhir, lebih disarankan memakai model empiris dari Script 13B.
AKTIFKAN_DENSITY_FALLBACK_JIKA_MODEL_MASSA_TIDAK_ADA = True
DENSITY_KG_PER_M3 = 850.0

# Jumlah frame untuk nilai rata-rata realtime dan capture pengujian.
ROLLING_AVG_FRAMES = 20
CAPTURE_AVG_FRAMES = 20

# Batas standar deviasi volume untuk status stabil. Nilai ini hanya indikator tampilan,
# bukan pengganti evaluasi akurasi terhadap timbangan digital.
STABLE_STD_VOLUME_CM3 = 20.0

# Threshold U-Net khusus untuk demo realtime.
# Jika mask batubara terlalu sedikit/muncul lemah, nilai 0.35-0.40 biasanya lebih longgar
# daripada 0.50. Nilai ini tidak otomatis mengikuti gambar, tetapi ditetapkan agar konsisten.
DEMO_UNET_THRESHOLD = 0.40

# Aturan keamanan hasil: volume/massa hanya dianggap valid jika ada piksel volume yang cukup.
# Jika valid_volume_pixels = 0, massa empiris tidak boleh ditampilkan sebagai hasil valid.
MIN_VALID_VOLUME_PIXELS = 50
MIN_VOLUME_CM3_VALID = 1.0

WINDOW_NAME = "DEMO REALTIME - ESTIMASI VOLUME DAN MASSA BATUBARA"
DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 720
DISPLAY_MIRROR_HORIZONTAL = True


# =========================================================
# IMPORT SCRIPT FINAL YANG SUDAH ADA
# =========================================================
def import_module_from_path(module_name, file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File pendukung tidak ditemukan:\n{file_path}\n"
            "Pastikan script ini diletakkan satu folder dengan Script 11 dan Script 12."
        )
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


s11 = import_module_from_path("script11_integrasi", SCRIPT11_PATH)
s12 = import_module_from_path("script12_volume", SCRIPT12_PATH)


# =========================================================
# UTILITAS FILE DAN MODEL MASSA
# =========================================================
def pastikan_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def baseline_key_from_value(value):
    try:
        val = float(value)
        if abs(val - round(val)) < 1e-6:
            return str(int(round(val)))
        return str(val).replace(".", "_")
    except Exception:
        return str(value)


def load_model_massa():
    """Load model kalibrasi empiris per baseline dari Script 13B."""
    if os.path.exists(WEIGHT_MODEL_JSON):
        with open(WEIGHT_MODEL_JSON, "r", encoding="utf-8") as f:
            obj = json.load(f)

        models = obj.get("models_by_baseline_cm", {})
        baseline_key = baseline_key_from_value(DEMO_BASELINE_CM)
        if baseline_key in models:
            m = models[baseline_key]
            a = float(m.get("a_gram_per_cm3", 0.0))
            b = float(m.get("b_gram", 0.0))
            print("[OK] Model kalibrasi massa per baseline dimuat")
            print(f"     {WEIGHT_MODEL_JSON}")
            print(f"     baseline demo = {DEMO_BASELINE_CM} cm")
            print(f"     massa_gram = {a:.8f} * volume_cm3 + {b:.8f}")
            return {
                "mode": "empiris_per_baseline",
                "baseline_cm": DEMO_BASELINE_CM,
                "a_gram_per_cm3": a,
                "b_gram": b,
                "source": WEIGHT_MODEL_JSON,
            }

        print(f"[WARNING] Model baseline {DEMO_BASELINE_CM} cm tidak ditemukan di JSON.")
        print(f"          Baseline tersedia: {list(models.keys())}")

    if AKTIFKAN_DENSITY_FALLBACK_JIKA_MODEL_MASSA_TIDAK_ADA:
        print("[WARNING] Model kalibrasi massa empiris per baseline belum tersedia.")
        print("          Massa akan dihitung sementara berdasarkan densitas.")
        print(f"          Densitas sementara = {DENSITY_KG_PER_M3:.2f} kg/m3")
        return {
            "mode": "densitas_sementara",
            "density_kg_per_m3": DENSITY_KG_PER_M3,
            "source": "fallback densitas sementara",
        }

    print("[WARNING] Model kalibrasi massa tidak ditemukan dan fallback densitas tidak aktif.")
    return None


def hitung_massa(volume_cm3, volume_m3, model_massa):
    if model_massa is None:
        return {
            "massa_gram": None,
            "massa_kg": None,
            "mode": "tidak_tersedia",
            "label": "MASSA: N/A",
        }

    if model_massa["mode"] in ["empiris", "empiris_per_baseline"]:
        raw_gram = model_massa["a_gram_per_cm3"] * volume_cm3 + model_massa["b_gram"]
        massa_gram = max(0.0, float(raw_gram))
        baseline_label = ""
        if model_massa["mode"] == "empiris_per_baseline":
            baseline_label = f" B{model_massa.get('baseline_cm', '')}cm"
        return {
            "massa_gram": massa_gram,
            "massa_kg": massa_gram / 1000.0,
            "mode": model_massa["mode"],
            "label": f"MASSA EMPIRIS{baseline_label}",
        }

    if model_massa["mode"] == "densitas_sementara":
        massa_kg = float(volume_m3) * float(model_massa["density_kg_per_m3"])
        return {
            "massa_gram": massa_kg * 1000.0,
            "massa_kg": massa_kg,
            "mode": "densitas_sementara",
            "label": "MASSA DENSITAS SEMENTARA",
        }

    return {
        "massa_gram": None,
        "massa_kg": None,
        "mode": "tidak_tersedia",
        "label": "MASSA: N/A",
    }


# =========================================================
# ROI GERBONG DAN HOMOGRAFI
# =========================================================
def load_roi_points_auto():
    """Memuat ROI lama tanpa prompt terminal agar cocok untuk demo realtime."""
    if not os.path.exists(s12.ROI_POINTS_JSON):
        return None
    try:
        with open(s12.ROI_POINTS_JSON, "r", encoding="utf-8") as f:
            obj = json.load(f)
        pts = np.asarray(obj.get("points", []), dtype=np.float32)
        if pts.shape == (4, 2):
            print(f"[OK] ROI gerbong dimuat: {s12.ROI_POINTS_JSON}")
            return s12.order_points(pts)
    except Exception as e:
        print(f"[WARNING] ROI lama gagal dibaca: {e}")
    return None


def pilih_roi_dari_frame(rect_left):
    """Pilih 4 titik ROI area dalam gerbong dari citra rectified kiri."""
    print("\n[INFO] ROI gerbong belum ada atau dipilih ulang.")
    print("       Klik 4 titik area DALAM gerbong:")
    print("       1 kiri-atas, 2 kanan-atas, 3 kanan-bawah, 4 kiri-bawah")
    print("       Tekan S untuk simpan, R untuk reset, ESC untuk batal.")

    img_show_base, scale = s12.resize_for_display(rect_left)
    points_display = []
    window_roi = "PILIH ROI GERBONG UNTUK DEMO REALTIME"

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points_display) < 4:
            points_display.append([x, y])

    cv2.namedWindow(window_roi, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_roi, img_show_base.shape[1], img_show_base.shape[0])
    cv2.setMouseCallback(window_roi, mouse_callback)

    while True:
        canvas = s12.gambar_instruksi_roi(img_show_base, points_display)
        cv2.imshow(window_roi, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            cv2.destroyWindow(window_roi)
            raise RuntimeError("Pemilihan ROI dibatalkan.")
        if key in [ord("r"), ord("R")]:
            points_display = []
            print("[INFO] Titik ROI direset.")
        if key in [ord("s"), ord("S")]:
            if len(points_display) != 4:
                print("[WARNING] Pilih 4 titik ROI terlebih dahulu.")
                continue
            pts = np.asarray(points_display, dtype=np.float32) / scale
            pts = s12.order_points(pts)
            s12.simpan_roi_points(pts)
            cv2.destroyWindow(window_roi)
            return pts


def buat_homografi_dari_roi(src_points):
    H, src_points_ordered, dst_points = s12.buat_homografi(src_points)
    return H, src_points_ordered, dst_points


# =========================================================
# PROSES FRAME REALTIME
# =========================================================
def ambil_frame_rectified_awal(capL, capR, stereomap):
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        raise RuntimeError("Kamera belum berhasil membaca frame untuk pemilihan ROI.")

    image_size = stereomap["image_size"]
    frameL = s11.siapkan_frame_input_realtime(frameL, image_size, s11.ROI_LEFT, "kiri")
    frameR = s11.siapkan_frame_input_realtime(frameR, image_size, s11.ROI_RIGHT, "kanan")

    rectL = cv2.remap(frameL, stereomap["mapLx"], stereomap["mapLy"], cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, stereomap["mapRx"], stereomap["mapRy"], cv2.INTER_LINEAR)
    return rectL, rectR


def colorize_volume_pixel(volume_pixel_cm3, valid_volume):
    arr = np.asarray(volume_pixel_cm3, dtype=np.float32)
    gray = np.zeros(arr.shape, dtype=np.uint8)
    if valid_volume.sum() > 0:
        max_val = max(float(np.percentile(arr[valid_volume], 95)), 1e-9)
        norm = np.clip(arr / max_val, 0, 1)
        gray = (norm * 255).astype(np.uint8)
        gray[~valid_volume] = 0
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid_volume] = (0, 0, 0)
    return color


def hitung_volume_demo(height_top_m, valid_volume):
    """Rumus sama dengan Script 12, tetapi tanpa print warning berulang saat realtime."""
    length_m = s12.GERBONG_LENGTH_CM / 100.0
    width_m = s12.GERBONG_WIDTH_CM / 100.0

    pixel_length_m = length_m / s12.TOPVIEW_W
    pixel_width_m = width_m / s12.TOPVIEW_H
    area_per_pixel_m2 = pixel_length_m * pixel_width_m

    volume_pixel_cm3 = np.zeros_like(height_top_m, dtype=np.float32)
    volume_pixel_cm3[valid_volume] = height_top_m[valid_volume] * area_per_pixel_m2 * 1_000_000.0

    volume_m3 = float(np.sum(height_top_m[valid_volume]) * area_per_pixel_m2)
    volume_cm3 = volume_m3 * 1_000_000.0
    volume_liter = volume_m3 * 1000.0
    valid_pixels = int(valid_volume.sum())

    if valid_pixels > 0:
        vals_cm = height_top_m[valid_volume] * 100.0
        mean_height_cm = float(np.mean(vals_cm))
        median_height_cm = float(np.median(vals_cm))
        max_height_cm = float(np.max(vals_cm))
    else:
        mean_height_cm = 0.0
        median_height_cm = 0.0
        max_height_cm = 0.0

    mask_pixels = int(valid_volume.size)
    valid_percent_topview = (valid_pixels / max(mask_pixels, 1)) * 100.0

    return {
        "area_per_pixel_m2": area_per_pixel_m2,
        "pixel_length_m": pixel_length_m,
        "pixel_width_m": pixel_width_m,
        "valid_pixels": valid_pixels,
        "valid_percent_topview": valid_percent_topview,
        "volume_m3": volume_m3,
        "volume_cm3": volume_cm3,
        "volume_liter": volume_liter,
        "mean_height_cm": mean_height_cm,
        "median_height_cm": median_height_cm,
        "max_height_cm": max_height_cm,
        "volume_pixel_cm3": volume_pixel_cm3,
    }


def proses_satu_frame(frameL, frameR, model_unet, stereomap, zref_m, floor_reference, matcher_bundle, H, model_massa):
    image_size = stereomap["image_size"]
    left_matcher, right_matcher, wls_filter, use_wls = matcher_bundle

    frameL = s11.siapkan_frame_input_realtime(frameL, image_size, s11.ROI_LEFT, "kiri")
    frameR = s11.siapkan_frame_input_realtime(frameR, image_size, s11.ROI_RIGHT, "kanan")

    rectL = cv2.remap(frameL, stereomap["mapLx"], stereomap["mapLy"], cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, stereomap["mapRx"], stereomap["mapRy"], cv2.INTER_LINEAR)

    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    disp_float = s11.hitung_disparity(grayL, grayR, left_matcher, right_matcher, wls_filter, use_wls)
    depth_m = s11.disparity_to_depth(disp_float, stereomap["Q"])
    valid_depth = s11.buat_valid_depth_mask(disp_float, depth_m)

    mask_unet = s11.prediksi_mask_unet(model_unet, rectL)
    height_m, valid_height = s11.hitung_height_map(depth_m, mask_unet, valid_depth, zref_m, floor_reference=floor_reference)
    height_stats = s11.statistik_height(height_m, valid_height, mask_unet)

    rect_top, height_top, valid_top, mask_top, valid_volume = s12.warp_topview(
        rectL, height_m, valid_height, mask_unet, H
    )
    volume_stats = hitung_volume_demo(height_top, valid_volume)

    # Aturan keamanan: massa baru dihitung jika volume benar-benar valid.
    valid_measurement = (
        int(volume_stats["valid_pixels"]) >= MIN_VALID_VOLUME_PIXELS
        and float(volume_stats["volume_cm3"]) >= MIN_VOLUME_CM3_VALID
    )

    if valid_measurement:
        massa_stats = hitung_massa(volume_stats["volume_cm3"], volume_stats["volume_m3"], model_massa)
    else:
        massa_stats = {
            "massa_gram": None,
            "massa_kg": None,
            "mode": "tidak_valid",
            "label": "MASSA TIDAK VALID",
            "reason": "valid volume pixels terlalu sedikit / volume hampir nol",
        }

    volume_color = colorize_volume_pixel(volume_stats["volume_pixel_cm3"], valid_volume)
    height_color = s12.colorize_height(height_top, valid_volume)
    mask_top_vis = cv2.cvtColor((mask_top.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    valid_vis = cv2.cvtColor((valid_volume.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    overlay_top = s12.overlay_mask(rect_top, mask_top)
    overlay_rect = s11.overlay_mask(rectL, mask_unet)

    return {
        "rectL": rectL,
        "rectR": rectR,
        "overlay_rect": overlay_rect,
        "mask_unet": mask_unet,
        "depth_m": depth_m,
        "valid_depth": valid_depth,
        "height_m": height_m,
        "valid_height": valid_height,
        "rect_top": rect_top,
        "height_top": height_top,
        "valid_top": valid_top,
        "mask_top": mask_top,
        "valid_volume": valid_volume,
        "height_color": height_color,
        "volume_color": volume_color,
        "mask_top_vis": mask_top_vis,
        "valid_vis": valid_vis,
        "overlay_top": overlay_top,
        "height_stats": height_stats,
        "volume_stats": volume_stats,
        "massa_stats": massa_stats,
        "valid_measurement": valid_measurement,
        "floor_reference_mode": "local_floor_map" if floor_reference is not None else "scalar_zref",
    }


# =========================================================
# STATISTIK ROLLING DAN DASHBOARD
# =========================================================
def update_rolling(rolling, result):
    # Rolling average hanya memakai frame yang valid agar nilai rata-rata tidak rusak
    # oleh frame kosong/salah mask.
    if not result.get("valid_measurement", False):
        return

    vs = result["volume_stats"]
    bs = result["massa_stats"]
    rolling.append({
        "volume_cm3": float(vs["volume_cm3"]),
        "volume_liter": float(vs["volume_liter"]),
        "volume_m3": float(vs["volume_m3"]),
        "mean_height_cm": float(vs["mean_height_cm"]),
        "valid_pixels": int(vs["valid_pixels"]),
        "massa_gram": None if bs["massa_gram"] is None else float(bs["massa_gram"]),
        "massa_kg": None if bs["massa_kg"] is None else float(bs["massa_kg"]),
    })


def ringkasan_rolling(rolling):
    if len(rolling) == 0:
        return None

    vol = np.array([r["volume_cm3"] for r in rolling], dtype=float)
    lit = np.array([r["volume_liter"] for r in rolling], dtype=float)
    m3 = np.array([r["volume_m3"] for r in rolling], dtype=float)
    hmean = np.array([r["mean_height_cm"] for r in rolling], dtype=float)
    valid_px = np.array([r["valid_pixels"] for r in rolling], dtype=float)

    massa_vals = [r["massa_gram"] for r in rolling if r["massa_gram"] is not None]
    if len(massa_vals) > 0:
        massa = np.array(massa_vals, dtype=float)
        massa_gram_mean = float(np.mean(massa))
        massa_gram_std = float(np.std(massa))
    else:
        massa_gram_mean = None
        massa_gram_std = None

    return {
        "n": len(rolling),
        "volume_cm3_mean": float(np.mean(vol)),
        "volume_cm3_std": float(np.std(vol)),
        "volume_liter_mean": float(np.mean(lit)),
        "volume_m3_mean": float(np.mean(m3)),
        "mean_height_cm_mean": float(np.mean(hmean)),
        "valid_pixels_mean": float(np.mean(valid_px)),
        "massa_gram_mean": massa_gram_mean,
        "massa_gram_std": massa_gram_std,
        "massa_kg_mean": None if massa_gram_mean is None else massa_gram_mean / 1000.0,
    }


def resize_dashboard_to_screen(dashboard):
    h, w = dashboard.shape[:2]
    scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
    if scale < 1.0:
        dashboard = cv2.resize(dashboard, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return dashboard


def mirror_untuk_tampilan(img):
    return cv2.flip(img, 1) if DISPLAY_MIRROR_HORIZONTAL else img


def teks_muatt_lebar(teks, font, scale, thickness, max_width):
    teks = str(teks)
    if cv2.getTextSize(teks, font, scale, thickness)[0][0] <= max_width:
        return teks
    suffix = "..."
    while len(teks) > 0:
        kandidat = teks[:-1].rstrip() + suffix
        if cv2.getTextSize(kandidat, font, scale, thickness)[0][0] <= max_width:
            return kandidat
        teks = teks[:-1]
    return suffix


def put_text_fit(canvas, teks, pos, font_scale, color, thickness=1, max_width=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    if max_width is None:
        max_width = canvas.shape[1] - x - 12
    teks = teks_muatt_lebar(teks, font, font_scale, thickness, max_width)
    cv2.putText(canvas, teks, (x, y), font, font_scale, color, thickness)


def gambar_sel_tabel(canvas, x, y, w, h, label, value, value_color=(0, 255, 255), bg=(28, 28, 28)):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (75, 75, 75), 1)
    put_text_fit(canvas, label, (x + 8, y + 15), 0.33, (170, 170, 170), 1, w - 16)
    put_text_fit(canvas, value, (x + 8, y + h - 10), 0.43, value_color, 1, w - 16)


def gambar_baris_sel(canvas, y, h, items, margin=12, gap=6):
    total_gap = gap * (len(items) - 1)
    cell_w = int((canvas.shape[1] - margin * 2 - total_gap) / len(items))
    x = margin
    for i, item in enumerate(items):
        w = cell_w if i < len(items) - 1 else canvas.shape[1] - margin - x
        gambar_sel_tabel(canvas, x, y, w, h, item["label"], item["value"], item.get("color", (0, 255, 255)), item.get("bg", (28, 28, 28)))
        x += w + gap


def buat_dashboard_demo(result, src_points, rolling_summary, fps, model_massa):
    rect_roi = s12.gambar_roi_pada_rectified(result["rectL"], src_points)

    p1 = s12.panel(mirror_untuk_tampilan(rect_roi), "LIVE RECTIFIED + ROI", 400, 220)
    p2 = s12.panel(mirror_untuk_tampilan(result["overlay_rect"]), "U-NET MASK PADA CITRA KIRI", 400, 220)
    p3 = s12.panel(mirror_untuk_tampilan(result["height_color"]), "HEIGHT MAP TOP-VIEW", 400, 220)
    p4 = s12.panel(mirror_untuk_tampilan(result["rect_top"]), "TOP-VIEW GERBONG", 400, 165)
    p5 = s12.panel(mirror_untuk_tampilan(result["volume_color"]), "VOLUME PER PIKSEL", 400, 165)
    p6 = s12.panel(mirror_untuk_tampilan(result["valid_vis"]), "PIKSEL VALID VOLUME", 400, 165)

    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])

    footer_h = 180
    footer = np.zeros((footer_h, row1.shape[1], 3), dtype=np.uint8)
    cv2.rectangle(footer, (0, 0), (footer.shape[1], footer.shape[0]), (18, 18, 18), -1)

    vs = result["volume_stats"]
    bs = result["massa_stats"]
    hs = result["height_stats"]
    valid_measurement = bool(result.get("valid_measurement", False))

    if rolling_summary is not None:
        volume_value = f"{rolling_summary['volume_cm3_mean']:.2f} cm3"
        liter_value = f"{rolling_summary['volume_liter_mean']:.4f} L"
        volume_std = f"{rolling_summary['volume_cm3_std']:.2f} cm3"
        if rolling_summary["massa_gram_mean"] is not None:
            massa_value = f"{rolling_summary['massa_gram_mean']:.2f} gram"
            massa_kg_value = f"{rolling_summary['massa_kg_mean']:.5f} kg"
            massa_std = f"{rolling_summary['massa_gram_std']:.2f} gram"
        else:
            massa_value = "belum tersedia"
            massa_kg_value = "-"
            massa_std = "-"
        status_stabil = "STABIL" if (valid_measurement and rolling_summary["volume_cm3_std"] <= STABLE_STD_VOLUME_CM3) else "TIDAK VALID" if not valid_measurement else "BELUM STABIL"
        mode_value = f"AVG {rolling_summary['n']} frame"
    else:
        volume_value = f"{vs['volume_cm3']:.2f} cm3"
        liter_value = f"{vs['volume_liter']:.4f} L"
        volume_std = "-"
        if bs["massa_gram"] is not None:
            massa_value = f"{bs['massa_gram']:.2f} gram"
            massa_kg_value = f"{bs['massa_kg']:.5f} kg"
            massa_std = "-"
        else:
            massa_value = "belum tersedia"
            massa_kg_value = "-"
            massa_std = "-"
        status_stabil = "VALID" if valid_measurement else "TIDAK VALID"
        mode_value = "SINGLE FRAME"

    warna_volume = (0, 255, 255) if valid_measurement else (0, 165, 255)
    warna_massa = (0, 230, 0) if valid_measurement else (0, 0, 255)

    gambar_baris_sel(footer, 8, 38, [
        {"label": "Status", "value": status_stabil, "color": warna_massa},
        {"label": "Mode", "value": mode_value, "color": (230, 230, 230)},
        {"label": "Volume", "value": volume_value, "color": warna_volume},
        {"label": "Liter", "value": liter_value, "color": warna_volume},
        {"label": "Volume Std", "value": volume_std, "color": (230, 230, 230)},
    ])

    gambar_baris_sel(footer, 50, 38, [
        {"label": bs["label"], "value": massa_value, "color": warna_massa},
        {"label": "Massa kg", "value": massa_kg_value, "color": warna_massa},
        {"label": "Massa Std", "value": massa_std, "color": (230, 230, 230)},
        {"label": "FPS", "value": f"{fps:.1f}", "color": (230, 230, 230)},
        {"label": "Valid Volume Px", "value": f"{vs['valid_pixels']}", "color": (230, 230, 230)},
    ])

    gambar_baris_sel(footer, 92, 38, [
        {"label": "Mean Height", "value": f"{vs['mean_height_cm']:.2f} cm"},
        {"label": "Median Height", "value": f"{vs['median_height_cm']:.2f} cm"},
        {"label": "Max Height", "value": f"{vs['max_height_cm']:.2f} cm"},
        {"label": "Valid Height", "value": f"{hs['valid_percent']:.1f}%"},
        {"label": "Area/Piksel", "value": f"{vs['area_per_pixel_m2']:.10f} m2"},
    ])

    gambar_baris_sel(footer, 134, 38, [
        {"label": "Rumus Volume", "value": "sum(h_i x A_piksel)", "color": (210, 210, 210)},
        {"label": "Kontrol", "value": "S=simpan | C=AVG | R=ROI ulang | ESC=keluar", "color": (210, 210, 210)},
    ])

    dashboard = np.vstack([row1, row2, footer])
    return resize_dashboard_to_screen(dashboard)


# =========================================================
# SIMPAN HASIL DEMO
# =========================================================
def append_history_csv(row):
    pastikan_output_dir()
    fieldnames = [
        "timestamp", "mode_simpan", "volume_cm3", "volume_liter", "volume_m3",
        "massa_gram", "massa_kg", "massa_mode", "mean_height_cm", "median_height_cm",
        "max_height_cm", "valid_pixels", "avg_frame_count", "std_volume_cm3",
        "status_stabil", "dashboard_path", "npz_path"
    ]
    exists = os.path.exists(OUTPUT_HISTORY_CSV)
    with open(OUTPUT_HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def simpan_hasil_demo(result, dashboard, src_points, dst_points, H, rolling_summary=None, mode_simpan="single"):
    pastikan_output_dir()
    waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"demo_realtime_{mode_simpan}_{waktu}"

    path_dashboard = os.path.join(OUTPUT_DIR, f"{prefix}_dashboard.png")
    path_rect = os.path.join(OUTPUT_DIR, f"{prefix}_rectified_left_roi.png")
    path_overlay_rect = os.path.join(OUTPUT_DIR, f"{prefix}_mask_overlay_rectified.png")
    path_top = os.path.join(OUTPUT_DIR, f"{prefix}_topview_gerbong.png")
    path_height = os.path.join(OUTPUT_DIR, f"{prefix}_height_topview.png")
    path_volume_pixel = os.path.join(OUTPUT_DIR, f"{prefix}_volume_per_pixel.png")
    path_valid = os.path.join(OUTPUT_DIR, f"{prefix}_valid_volume_pixels.png")
    path_npz = os.path.join(OUTPUT_DIR, f"{prefix}_data.npz")
    path_txt = os.path.join(OUTPUT_DIR, f"{prefix}_ringkasan.txt")

    rect_roi = s12.gambar_roi_pada_rectified(result["rectL"], src_points)
    cv2.imwrite(path_dashboard, dashboard)
    cv2.imwrite(path_rect, rect_roi)
    cv2.imwrite(path_overlay_rect, result["overlay_rect"])
    cv2.imwrite(path_top, result["rect_top"])
    cv2.imwrite(path_height, result["height_color"])
    cv2.imwrite(path_volume_pixel, result["volume_color"])
    cv2.imwrite(path_valid, result["valid_vis"])

    vs = result["volume_stats"]
    bs = result["massa_stats"]

    if rolling_summary is not None:
        save_volume_cm3 = rolling_summary["volume_cm3_mean"]
        save_volume_liter = rolling_summary["volume_liter_mean"]
        save_volume_m3 = rolling_summary["volume_m3_mean"]
        save_massa_gram = rolling_summary["massa_gram_mean"]
        save_massa_kg = rolling_summary["massa_kg_mean"]
        avg_n = rolling_summary["n"]
        std_volume = rolling_summary["volume_cm3_std"]
        status_stabil = "STABIL" if std_volume <= STABLE_STD_VOLUME_CM3 else "BELUM_STABIL"
    else:
        save_volume_cm3 = vs["volume_cm3"]
        save_volume_liter = vs["volume_liter"]
        save_volume_m3 = vs["volume_m3"]
        save_massa_gram = bs["massa_gram"]
        save_massa_kg = bs["massa_kg"]
        avg_n = 1
        std_volume = 0.0
        status_stabil = "SINGLE_FRAME"

    np.savez_compressed(
        path_npz,
        H=H,
        src_points=src_points,
        dst_points=dst_points,
        rectified_left=result["rectL"],
        rectified_right=result["rectR"],
        mask_unet=result["mask_unet"].astype(np.uint8),
        depth_m=result["depth_m"].astype(np.float32),
        valid_depth=result["valid_depth"].astype(np.uint8),
        height_m=result["height_m"].astype(np.float32),
        valid_height=result["valid_height"].astype(np.uint8),
        rect_top=result["rect_top"],
        height_top_m=result["height_top"].astype(np.float32),
        mask_top=result["mask_top"].astype(np.uint8),
        valid_volume=result["valid_volume"].astype(np.uint8),
        volume_pixel_cm3=vs["volume_pixel_cm3"].astype(np.float32),
        area_per_pixel_m2=np.array([vs["area_per_pixel_m2"]], dtype=np.float32),
        volume_cm3=np.array([save_volume_cm3], dtype=np.float32),
        volume_liter=np.array([save_volume_liter], dtype=np.float32),
        volume_m3=np.array([save_volume_m3], dtype=np.float32),
        massa_gram=np.array([-1.0 if save_massa_gram is None else save_massa_gram], dtype=np.float32),
        massa_kg=np.array([-1.0 if save_massa_kg is None else save_massa_kg], dtype=np.float32),
        avg_frame_count=np.array([avg_n], dtype=np.int32),
        std_volume_cm3=np.array([std_volume], dtype=np.float32),
        floor_reference_mode=np.array([result.get("floor_reference_mode", "unknown")]),
    )

    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("RINGKASAN DEMO REALTIME SISTEM ESTIMASI VOLUME DAN MASSA BATUBARA\n")
        f.write("===============================================================\n")
        f.write(f"Waktu pengujian       : {waktu}\n")
        f.write(f"Mode simpan           : {mode_simpan}\n")
        f.write(f"Jumlah frame rata-rata: {avg_n}\n")
        f.write(f"Volume cm3            : {save_volume_cm3:.6f}\n")
        f.write(f"Volume liter          : {save_volume_liter:.6f}\n")
        f.write(f"Volume m3             : {save_volume_m3:.10f}\n")
        if save_massa_gram is not None:
            f.write(f"Massa gram            : {save_massa_gram:.6f}\n")
            f.write(f"Massa kg              : {save_massa_kg:.8f}\n")
        else:
            f.write("Massa                 : tidak tersedia\n")
        f.write(f"Mode massa            : {bs['mode']}\n")
        f.write(f"Floor reference mode  : {result.get('floor_reference_mode', 'unknown')}\n")
        f.write(f"Mean height cm        : {vs['mean_height_cm']:.6f}\n")
        f.write(f"Median height cm      : {vs['median_height_cm']:.6f}\n")
        f.write(f"Max height cm         : {vs['max_height_cm']:.6f}\n")
        f.write(f"Valid volume pixels   : {vs['valid_pixels']}\n")
        f.write(f"Std volume cm3        : {std_volume:.6f}\n")
        f.write(f"Status stabil         : {status_stabil}\n")
        f.write("\nRumus volume:\n")
        f.write("V_total = sum(h_i x A_piksel)\n")
        f.write("h_i = tinggi batubara pada piksel ke-i dari height map\n")
        f.write("A_piksel = luas nyata satu piksel pada top-view hasil homografi\n")
        f.write(f"A_piksel m2           : {vs['area_per_pixel_m2']:.10f}\n")
        f.write("\nCatatan:\n")
        f.write("Nilai realtime dapat berubah antar-frame karena noise depth, pencahayaan, dan hasil segmentasi.\n")
        f.write("Untuk validasi akhir, bandingkan massa estimasi dengan timbangan digital.\n")

    append_history_csv({
        "timestamp": waktu,
        "mode_simpan": mode_simpan,
        "volume_cm3": save_volume_cm3,
        "volume_liter": save_volume_liter,
        "volume_m3": save_volume_m3,
        "massa_gram": "" if save_massa_gram is None else save_massa_gram,
        "massa_kg": "" if save_massa_kg is None else save_massa_kg,
        "massa_mode": bs["mode"],
        "mean_height_cm": vs["mean_height_cm"],
        "median_height_cm": vs["median_height_cm"],
        "max_height_cm": vs["max_height_cm"],
        "valid_pixels": vs["valid_pixels"],
        "avg_frame_count": avg_n,
        "std_volume_cm3": std_volume,
        "status_stabil": status_stabil,
        "dashboard_path": path_dashboard,
        "npz_path": path_npz,
    })

    print("\n[OK] Hasil demo realtime tersimpan:")
    print(f"     Dashboard : {path_dashboard}")
    print(f"     Data NPZ  : {path_npz}")
    print(f"     Ringkasan : {path_txt}")
    print(f"     CSV       : {OUTPUT_HISTORY_CSV}")


# =========================================================
# CAPTURE RATA-RATA BEBERAPA FRAME
# =========================================================
def capture_average_frames(capL, capR, model_unet, stereomap, zref_m, floor_reference, matcher_bundle, H, model_massa, src_points, dst_points):
    print(f"\n[INFO] Capture AVG dimulai: mengambil {CAPTURE_AVG_FRAMES} frame...")
    temp_rolling = deque(maxlen=CAPTURE_AVG_FRAMES)
    last_result = None
    last_dashboard = None
    t_prev = time.time()

    for i in range(CAPTURE_AVG_FRAMES):
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print(f"[WARNING] Frame capture ke-{i+1} gagal dibaca.")
            continue

        result = proses_satu_frame(frameL, frameR, model_unet, stereomap, zref_m, floor_reference, matcher_bundle, H, model_massa)
        update_rolling(temp_rolling, result)
        summary = ringkasan_rolling(temp_rolling)

        now = time.time()
        fps = 1.0 / max(now - t_prev, 1e-9)
        t_prev = now

        dashboard = buat_dashboard_demo(result, src_points, summary, fps, model_massa)
        cv2.imshow(WINDOW_NAME, dashboard)
        cv2.waitKey(1)

        last_result = result
        last_dashboard = dashboard

    summary = ringkasan_rolling(temp_rolling)
    if last_result is None:
        print("[WARNING] Capture AVG gagal karena tidak ada frame terbaca.")
        return
    if summary is None:
        print("[WARNING] Capture AVG selesai, tetapi tidak ada frame valid untuk volume/massa.")
        print("          Periksa mask U-Net, ROI, depth map, dan threshold.")

    simpan_hasil_demo(last_result, last_dashboard, src_points, dst_points, H, rolling_summary=summary, mode_simpan="avg")
    print("[OK] Capture AVG selesai.")


# =========================================================
# MAIN
# =========================================================
def main():
    pastikan_output_dir()

    print("=" * 72)
    print("SCRIPT 16 — DEMO REALTIME VOLUME DAN MASSA BATUBARA PER BASELINE")
    print("=" * 72)
    print(f"Script 11 : {SCRIPT11_PATH}")
    print(f"Script 12 : {SCRIPT12_PATH}")
    print(f"Output    : {OUTPUT_DIR}")
    print(f"ROI crop  : kiri={s11.ROI_LEFT} | kanan={s11.ROI_RIGHT}")
    print(f"Gerbong   : {s12.GERBONG_LENGTH_CM:.1f} cm x {s12.GERBONG_WIDTH_CM:.1f} cm x {s12.GERBONG_HEIGHT_CM:.1f} cm")

    print("\n[1] Load model U-Net, stereomap, Z_ref, dan model massa...")
    model_unet = s11.load_model_unet()
    s11.UNET_THRESHOLD = DEMO_UNET_THRESHOLD
    print(f"[INFO] Threshold U-Net demo dipakai: {s11.UNET_THRESHOLD:.2f}")

    stereomap = s11.load_stereomap()
    zref_m = s11.load_zref()
    floor_reference = s11.load_referensi_lantai_lokal(stereomap["image_size"], zref_m)
    model_massa = load_model_massa()

    print("\n[2] Buka kamera stereo...")
    capL, capR = s11.buka_dua_kamera(stereomap["image_size"])

    print("\n[3] Siapkan SGBM/WLS...")
    matcher_bundle = s11.buat_sgbm()

    print("\n[4] Siapkan ROI homografi gerbong...")
    rectL_awal, _ = ambil_frame_rectified_awal(capL, capR, stereomap)
    src_points = load_roi_points_auto()
    if src_points is None:
        src_points = pilih_roi_dari_frame(rectL_awal)
    H, src_points, dst_points = buat_homografi_dari_roi(src_points)

    print("\n[5] Demo realtime aktif.")
    print("    S   = simpan frame saat ini")
    print("    C   = capture AVG beberapa frame")
    print("    R   = pilih ulang ROI gerbong")
    print("    ESC = keluar")

    rolling = deque(maxlen=ROLLING_AVG_FRAMES)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_MAX_W, DISPLAY_MAX_H)

    prev_time = time.time()
    last_result = None
    last_dashboard = None

    try:
        while True:
            retL, frameL = capL.read()
            retR, frameR = capR.read()
            if not retL or not retR:
                print("[WARNING] Frame kamera gagal dibaca. Cek koneksi kamera.")
                time.sleep(0.05)
                continue

            result = proses_satu_frame(frameL, frameR, model_unet, stereomap, zref_m, floor_reference, matcher_bundle, H, model_massa)
            update_rolling(rolling, result)
            summary = ringkasan_rolling(rolling)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now

            dashboard = buat_dashboard_demo(result, src_points, summary, fps, model_massa)
            cv2.imshow(WINDOW_NAME, dashboard)

            last_result = result
            last_dashboard = dashboard

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key in [ord("s"), ord("S")]:
                simpan_hasil_demo(last_result, last_dashboard, src_points, dst_points, H, rolling_summary=None, mode_simpan="single")
            elif key in [ord("c"), ord("C")]:
                capture_average_frames(capL, capR, model_unet, stereomap, zref_m, floor_reference, matcher_bundle, H, model_massa, src_points, dst_points)
            elif key in [ord("r"), ord("R")]:
                src_points = pilih_roi_dari_frame(last_result["rectL"])
                H, src_points, dst_points = buat_homografi_dari_roi(src_points)
                rolling.clear()
                print("[OK] ROI diperbarui. Rolling average direset.")

    finally:
        capL.release()
        capR.release()
        cv2.destroyAllWindows()
        print("\n[SELESAI] Demo realtime ditutup.")
        print(f"Output tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
