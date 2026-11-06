Kontrol pemilihan ROI:
- Klik 4 titik urut: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah.
- R = reset titik.
- S = simpan ROI setelah 4 titik dipilih.
- ESC = batal.
"""

import os
import json
import csv
from glob import glob
from datetime import datetime

import cv2
import numpy as np


# =========================================================
# KONFIGURASI PATH
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "hasil_integrasi_unet_depth")
OUTPUT_DIR = os.path.join(BASE_DIR, "hasil_volume_homografi")

# MODE:
# "single" = proses 1 snapshot terbaru saja.
# "batch"  = proses semua snapshot Script 11 secara berurutan.
MODE = "batch"

# Untuk MODE single: jika None, script otomatis mengambil snapshot terbaru *_data.npz dari INPUT_DIR.
INPUT_NPZ_PATH = None

# Kalau batch, urutkan dari data lama ke data baru agar cocok untuk data kalibrasi.
URUTKAN_BATCH_DARI_LAMA_KE_BARU = True

# Supaya dataset kalibrasi massa rapi: setiap batch membangun ulang volume_history.csv
# dari semua snapshot Script 11, bukan menumpuk data lama yang bisa duplikat.
OVERWRITE_VOLUME_HISTORY_ON_BATCH = True

# File penyimpanan 4 titik ROI gerbong, supaya tidak perlu klik ulang setiap run.
ROI_POINTS_JSON = os.path.join(OUTPUT_DIR, "roi_gerbong_points.json")

# Deteksi otomatis ROI lantai dalam gerbong.
# Tetap konservatif: jika confidence rendah, script fallback ke ROI lama / klik manual.
AUTO_DETECT_ROI = True
AUTO_ROI_MIN_CONFIDENCE = 0.62
AUTO_ROI_SHRINK_TO_INNER = 0.92
AUTO_ROI_SAVE_DEBUG = True
AUTO_ROI_CONFIRM_BEFORE_USE = False


# =========================================================
# DIMENSI GERBONG MINIATUR
# =========================================================
GERBONG_LENGTH_CM = 40.0
GERBONG_WIDTH_CM = 10.0
GERBONG_HEIGHT_CM = 7.5

# Resolusi top-view. 10 px/cm -> 40 cm = 400 px, 10 cm = 100 px.
TOPVIEW_PIXELS_PER_CM = 10
TOPVIEW_W = int(round(GERBONG_LENGTH_CM * TOPVIEW_PIXELS_PER_CM))
TOPVIEW_H = int(round(GERBONG_WIDTH_CM * TOPVIEW_PIXELS_PER_CM))

# Batas tinggi fisik. Diberi margin 20% dari tinggi gerbong untuk toleransi mound/noise.
HEIGHT_MIN_M = 0.0
HEIGHT_MAX_M = 12.0 / 100.0

# Minimum luas mask valid agar hasil tidak dihitung dari noise kecil.
MIN_VALID_AREA_PIXELS = 20

# Estimasi massa opsional. Default False agar volume dulu yang dipastikan benar.
AKTIFKAN_ESTIMASI_MASSA = False
DENSITY_KG_PER_M3 = 850.0  # nilai contoh; sebaiknya diganti hasil kalibrasi/rujukan material


# =========================================================
# TAMPILAN
# =========================================================
DISPLAY_MAX_W = 1100
DISPLAY_MAX_H = 620
WINDOW_ROI = "Pilih 4 Sudut ROI Gerbong [S=Simpan | R=Reset | ESC=Batal]"


# =========================================================
# UTILITAS FILE
# =========================================================
def pastikan_folder_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def cari_npz_terbaru():
    if INPUT_NPZ_PATH is not None:
        if not os.path.exists(INPUT_NPZ_PATH):
            raise FileNotFoundError(f"INPUT_NPZ_PATH tidak ditemukan: {INPUT_NPZ_PATH}")
        return INPUT_NPZ_PATH

    pattern = os.path.join(INPUT_DIR, "*_data.npz")
    files = sorted(glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        raise FileNotFoundError(
            "Tidak ada file *_data.npz dari Script 11.\n"
            f"Folder dicek: {INPUT_DIR}\n"
            "Jalankan Script 11 terlebih dahulu, lalu tekan tombol S untuk menyimpan snapshot."
        )
    return files[0]


def cari_semua_npz():
    pattern = os.path.join(INPUT_DIR, "*_data.npz")
    files = glob(pattern)
    if not files:
        raise FileNotFoundError(
            "Tidak ada file *_data.npz dari Script 11.\n"
            f"Folder dicek: {INPUT_DIR}\n"
            "Jalankan Script 11 terlebih dahulu, lalu tekan S untuk menyimpan snapshot."
        )
    files = sorted(files, key=os.path.getmtime, reverse=not URUTKAN_BATCH_DARI_LAMA_KE_BARU)
    return files


def prefix_dari_npz(npz_path):
    name = os.path.basename(npz_path)
    if name.endswith("_data.npz"):
        return name[:-len("_data.npz")]
    return os.path.splitext(name)[0]


def cari_rectified_left(npz_path):
    prefix = prefix_dari_npz(npz_path)
    kandidat = [
        os.path.join(os.path.dirname(npz_path), f"{prefix}_rectified_left.png"),
        os.path.join(os.path.dirname(npz_path), f"{prefix}_rectL.png"),
        os.path.join(os.path.dirname(npz_path), f"{prefix}_left.png"),
    ]
    for p in kandidat:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "File rectified_left untuk snapshot ini tidak ditemukan. Dicari:\n"
        + "\n".join(kandidat)
    )


def baca_scalar_npz(data, key, default=None):
    if key not in data:
        return default
    try:
        arr = np.asarray(data[key]).astype(float).ravel()
        if arr.size == 0:
            return default
        val = float(arr[0])
        if np.isnan(val):
            return default
        return val
    except Exception:
        return default


def baca_int_npz(data, key, default=None):
    val = baca_scalar_npz(data, key, default=None)
    if val is None:
        return default
    try:
        return int(round(float(val)))
    except Exception:
        return default


def baca_string_npz(data, key, default=""):
    if key not in data:
        return default
    try:
        arr = np.asarray(data[key]).ravel()
        if arr.size == 0:
            return default
        return str(arr[0])
    except Exception:
        return default


def load_snapshot(npz_path=None):
    if npz_path is None:
        npz_path = cari_npz_terbaru()
    img_path = cari_rectified_left(npz_path)
    data = np.load(npz_path)

    required = ["height_m", "valid_height", "mask_unet"]
    for key in required:
        if key not in data:
            raise KeyError(f"Data '{key}' tidak ditemukan di NPZ: {npz_path}")

    rect_left = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if rect_left is None:
        raise RuntimeError(f"Gagal membaca rectified_left: {img_path}")

    height_m = data["height_m"].astype(np.float32)
    valid_height = data["valid_height"].astype(np.uint8) > 0
    mask_unet = data["mask_unet"].astype(np.uint8) > 0
    depth_m = data["depth_m"].astype(np.float32) if "depth_m" in data else None
    zref_m = baca_scalar_npz(data, "zref_m", default=np.nan)
    floor_reference_mode = baca_string_npz(data, "floor_reference_mode", default="unknown")
    floor_reference_path = baca_string_npz(data, "floor_reference_path", default="")

    # Metadata dari Script 11 versi kalibrasi massa. Jika file lama belum punya metadata,
    # nilai dibiarkan kosong agar tetap kompatibel.
    sample_id = baca_string_npz(data, "sample_id", default=prefix_dari_npz(npz_path))
    baseline_cm = baca_scalar_npz(data, "baseline_cm", default=None)
    target_weight_g = baca_scalar_npz(data, "target_weight_g", default=None)
    actual_weight_g = baca_scalar_npz(data, "actual_weight_g", default=None)
    condition = baca_string_npz(data, "condition", default="")
    profile = baca_string_npz(data, "profile", default="")
    repetition = baca_int_npz(data, "repetition", default=None)
    notes = baca_string_npz(data, "notes", default="")

    # Samakan ukuran kalau ada beda kecil.
    h, w = rect_left.shape[:2]
    if height_m.shape[:2] != (h, w):
        height_m = cv2.resize(height_m, (w, h), interpolation=cv2.INTER_LINEAR)
    if valid_height.shape[:2] != (h, w):
        valid_height = cv2.resize(valid_height.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    if mask_unet.shape[:2] != (h, w):
        mask_unet = cv2.resize(mask_unet.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    if depth_m is not None and depth_m.shape[:2] != (h, w):
        depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_LINEAR)

    prefix = prefix_dari_npz(npz_path)
    print("[OK] Snapshot dimuat")
    print(f"     NPZ        : {npz_path}")
    print(f"     Rect left  : {img_path}")
    print(f"     Prefix     : {prefix}")
    print(f"     Sample ID  : {sample_id}")
    if baseline_cm is not None or target_weight_g is not None:
        print(f"     Metadata   : baseline={baseline_cm} cm | target={target_weight_g} g | profile={profile} | rep={repetition}")
    print(f"     Image size : {w} x {h}")
    if np.isfinite(zref_m):
        print(f"     Z_ref      : {zref_m:.4f} m | {zref_m*100:.2f} cm")
    print(f"     Floor ref  : {floor_reference_mode} | {floor_reference_path}")

    return {
        "npz_path": npz_path,
        "img_path": img_path,
        "prefix": prefix,
        "sample_id": sample_id,
        "baseline_cm": baseline_cm,
        "target_weight_g": target_weight_g,
        "actual_weight_g": actual_weight_g,
        "condition": condition,
        "profile": profile,
        "repetition": repetition,
        "notes": notes,
        "rect_left": rect_left,
        "height_m": height_m,
        "valid_height": valid_height,
        "mask_unet": mask_unet,
        "depth_m": depth_m,
        "zref_m": zref_m,
        "floor_reference_mode": floor_reference_mode,
        "floor_reference_path": floor_reference_path,
    }




def csv_value(value):
    if value is None:
        return ""
    try:
        if isinstance(value, float) and np.isnan(value):
            return ""
    except Exception:
        pass
    return value

# =========================================================
# ROI / HOMOGRAFI
# =========================================================
def order_points(pts):
    """Urutkan titik menjadi: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left
    return ordered


def resize_for_display(img):
    h, w = img.shape[:2]
    scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale
    return img.copy(), 1.0


def simpan_roi_points(points):
    pastikan_folder_output()
    obj = {
        "points_order": "top_left, top_right, bottom_right, bottom_left",
        "points": np.asarray(points, dtype=float).tolist(),
        "gerbong_length_cm": GERBONG_LENGTH_CM,
        "gerbong_width_cm": GERBONG_WIDTH_CM,
        "topview_w": TOPVIEW_W,
        "topview_h": TOPVIEW_H,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(ROI_POINTS_JSON, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"[OK] ROI disimpan: {ROI_POINTS_JSON}")


def load_roi_points():
    if not os.path.exists(ROI_POINTS_JSON):
        return None
    with open(ROI_POINTS_JSON, "r", encoding="utf-8") as f:
        obj = json.load(f)
    pts = np.asarray(obj.get("points", []), dtype=np.float32)
    if pts.shape != (4, 2):
        return None
    print(f"[OK] ROI lama dimuat: {ROI_POINTS_JSON}")
    print("     Jika ingin pilih ulang, hapus file roi_gerbong_points.json atau tekan N saat diminta.")
    return pts


def polygon_area_abs(points):
    return abs(float(cv2.contourArea(np.asarray(points, dtype=np.float32))))


def shrink_quad_to_inner(points, scale=AUTO_ROI_SHRINK_TO_INNER):
    pts = order_points(points).astype(np.float32)
    center = np.mean(pts, axis=0)
    return (center + (pts - center) * float(scale)).astype(np.float32)


def score_roi_candidate(points, image_shape):
    pts = order_points(points).astype(np.float32)
    h, w = image_shape[:2]
    area = polygon_area_abs(pts)
    frame_area = float(max(1, h * w))
    area_ratio = area / frame_area
    if area_ratio < 0.015 or area_ratio > 0.75:
        return 0.0, {}
    if not cv2.isContourConvex(pts.astype(np.int32)):
        return 0.0, {}

    top_w = float(np.linalg.norm(pts[1] - pts[0]))
    bot_w = float(np.linalg.norm(pts[2] - pts[3]))
    left_h = float(np.linalg.norm(pts[3] - pts[0]))
    right_h = float(np.linalg.norm(pts[2] - pts[1]))
    mean_w = max(1e-6, (top_w + bot_w) / 2.0)
    mean_h = max(1e-6, (left_h + right_h) / 2.0)
    aspect = mean_w / mean_h
    expected_aspect = GERBONG_LENGTH_CM / GERBONG_WIDTH_CM

    aspect_score = max(0.0, 1.0 - abs(aspect - expected_aspect) / expected_aspect)
    parallel_score = max(0.0, 1.0 - (abs(top_w - bot_w) / mean_w + abs(left_h - right_h) / mean_h) / 2.0)
    center = np.mean(pts, axis=0)
    center_dist = np.linalg.norm(center - np.array([w / 2.0, h / 2.0], dtype=np.float32))
    center_score = max(0.0, 1.0 - center_dist / (0.75 * max(w, h)))
    area_score = min(1.0, area_ratio / 0.12)

    score = 0.42 * aspect_score + 0.25 * parallel_score + 0.18 * center_score + 0.15 * area_score
    info = {
        "score": score,
        "aspect": aspect,
        "area_ratio": area_ratio,
        "aspect_score": aspect_score,
        "parallel_score": parallel_score,
        "center_score": center_score,
        "area_score": area_score,
    }
    return float(score), info


def candidate_quads_from_edges(rect_left):
    gray = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    edges = cv2.Canny(gray, 45, 135)
    kernel = np.ones((5, 5), dtype=np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    quads = []
    frame_area = rect_left.shape[0] * rect_left.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.01:
            continue
        peri = cv2.arcLength(contour, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.025 * peri, True)
        if len(approx) == 4:
            quads.append(approx.reshape(4, 2).astype(np.float32))

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        if polygon_area_abs(box) >= frame_area * 0.01:
            quads.append(box)
    return quads


def simpan_debug_roi_otomatis(rect_left, points, score, info):
    if not AUTO_ROI_SAVE_DEBUG:
        return ""
    pastikan_folder_output()
    path = os.path.join(OUTPUT_DIR, "roi_gerbong_auto_debug.png")
    out = rect_left.copy()
    pts = order_points(points).astype(np.int32)
    cv2.polylines(out, [pts.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
    for i, p in enumerate(pts):
        cv2.circle(out, tuple(p), 7, (0, 0, 255), -1)
        cv2.putText(out, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    label = f"auto ROI score={score:.3f} aspect={info.get('aspect', 0.0):.2f}"
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 620), 42), (0, 0, 0), -1)
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
    cv2.imwrite(path, out)
    return path


def deteksi_roi_gerbong_otomatis(rect_left):
    candidates = candidate_quads_from_edges(rect_left)
    best_points = None
    best_score = 0.0
    best_info = {}

    for cand in candidates:
        inner = shrink_quad_to_inner(cand)
        score, info = score_roi_candidate(inner, rect_left.shape)
        if score > best_score:
            best_points = inner
            best_score = score
            best_info = info

    if best_points is None:
        return None, 0.0, {}, ""

    debug_path = simpan_debug_roi_otomatis(rect_left, best_points, best_score, best_info)
    return order_points(best_points), best_score, best_info, debug_path


def gambar_instruksi_roi(img, points):
    """Tampilan pemilihan ROI yang lebih rapi, teks tidak terpotong."""
    out = img.copy()

    # Bar instruksi dibuat lebih tinggi dan teks dipisah menjadi beberapa baris
    header_h = 118
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], header_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.78, out, 0.22, 0)

    teks1 = "Klik 4 titik ROI area DALAM gerbong"
    teks2 = "Urutan: 1 kiri-atas | 2 kanan-atas | 3 kanan-bawah | 4 kiri-bawah"
    teks3 = "S = Simpan   |   R = Reset   |   ESC = Batal"

    # Font diperkecil supaya semua teks tampil penuh di layar laptop
    cv2.putText(out, teks1, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(out, teks2, (14, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
    cv2.putText(out, teks3, (14, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

    # Gambar titik dan garis
    for i, p in enumerate(points):
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(out, (x, y), 7, (0, 255, 255), -1)
        cv2.circle(out, (x, y), 10, (0, 0, 0), 2)
        cv2.putText(out, str(i + 1), (x + 12, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)

    if len(points) >= 2:
        pts = np.asarray(points, dtype=np.int32)
        for i in range(len(points) - 1):
            cv2.line(out, tuple(pts[i]), tuple(pts[i + 1]), (0, 255, 255), 3)
        if len(points) == 4:
            cv2.line(out, tuple(pts[3]), tuple(pts[0]), (0, 255, 255), 3)

            # Transparansi area ROI supaya bentuk gerbong terlihat jelas
            poly = pts.reshape((-1, 1, 2))
            roi_overlay = out.copy()
            cv2.fillPoly(roi_overlay, [poly], (0, 255, 255))
            out = cv2.addWeighted(roi_overlay, 0.18, out, 0.82, 0)
            cv2.polylines(out, [poly], True, (0, 255, 255), 3)

    return out

def pilih_roi_interaktif(rect_left):
    existing = load_roi_points()
    if existing is not None:
        print("\nGunakan ROI lama? [Y/n]")
        ans = input("Jawaban: ").strip().lower()
        if ans in ["", "y", "yes", "ya"]:
            return existing

    img_show_base, scale = resize_for_display(rect_left)
    points_display = []

    print("\n[INFO] Tampilan ROI sudah dikecilkan agar muat di layar.")
    print("       Klik titik di area DALAM gerbong, bukan bagian luar/dinding luar.")

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points_display) < 4:
            points_display.append([x, y])

    cv2.namedWindow(WINDOW_ROI, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_ROI, img_show_base.shape[1], img_show_base.shape[0])
    cv2.setMouseCallback(WINDOW_ROI, mouse_callback)

    while True:
        canvas = gambar_instruksi_roi(img_show_base, points_display)
        cv2.imshow(WINDOW_ROI, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(WINDOW_ROI)
            raise RuntimeError("Pemilihan ROI dibatalkan.")
        elif key in [ord("r"), ord("R")]:
            points_display = []
            print("[INFO] Titik ROI direset.")
        elif key in [ord("s"), ord("S")]:
            if len(points_display) != 4:
                print("[WARNING] Pilih 4 titik dulu sebelum menyimpan ROI.")
                continue
            pts = np.asarray(points_display, dtype=np.float32) / scale
            pts = order_points(pts)
            simpan_roi_points(pts)
            cv2.destroyWindow(WINDOW_ROI)
            return pts


# =========================================================
# VOLUME
# =========================================================
def buat_homografi(src_points):
    src = order_points(src_points).astype(np.float32)
    dst = np.array(
        [
            [0, 0],
            [TOPVIEW_W - 1, 0],
            [TOPVIEW_W - 1, TOPVIEW_H - 1],
            [0, TOPVIEW_H - 1],
        ],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    return H, src, dst


def warp_topview(rect_left, height_m, valid_height, mask_unet, H):
    size = (TOPVIEW_W, TOPVIEW_H)
    rect_top = cv2.warpPerspective(rect_left, H, size, flags=cv2.INTER_LINEAR)
    height_top = cv2.warpPerspective(height_m.astype(np.float32), H, size, flags=cv2.INTER_LINEAR)
    valid_top = cv2.warpPerspective(valid_height.astype(np.uint8), H, size, flags=cv2.INTER_NEAREST) > 0
    mask_top = cv2.warpPerspective(mask_unet.astype(np.uint8), H, size, flags=cv2.INTER_NEAREST) > 0

    # Bersihkan nilai tinggi tidak masuk akal
    height_top = np.nan_to_num(height_top, nan=0.0, posinf=0.0, neginf=0.0)
    valid_volume = mask_top & valid_top
    valid_volume &= height_top > HEIGHT_MIN_M
    valid_volume &= height_top <= HEIGHT_MAX_M

    height_top_clean = np.zeros_like(height_top, dtype=np.float32)
    height_top_clean[valid_volume] = height_top[valid_volume]

    return rect_top, height_top_clean, valid_top, mask_top, valid_volume


def hitung_volume(height_top_m, valid_volume):
    # ukuran real top-view dalam meter
    length_m = GERBONG_LENGTH_CM / 100.0
    width_m = GERBONG_WIDTH_CM / 100.0

    pixel_length_m = length_m / TOPVIEW_W
    pixel_width_m = width_m / TOPVIEW_H
    area_per_pixel_m2 = pixel_length_m * pixel_width_m
    area_per_pixel_cm2 = area_per_pixel_m2 * 10000.0

    if valid_volume.sum() < MIN_VALID_AREA_PIXELS:
        print("[WARNING] Piksel valid untuk volume sangat sedikit. Cek mask, depth, dan ROI.")

    volume_m3 = float(np.sum(height_top_m[valid_volume]) * area_per_pixel_m2)
    volume_cm3 = volume_m3 * 1_000_000.0
    volume_liter = volume_m3 * 1000.0

    valid_pixels = int(valid_volume.sum())
    mean_height_cm = 0.0
    median_height_cm = 0.0
    p95_height_cm = 0.0
    p98_height_cm = 0.0
    max_height_cm = 0.0
    if valid_pixels > 0:
        vals_cm = height_top_m[valid_volume] * 100.0
        mean_height_cm = float(np.mean(vals_cm))
        median_height_cm = float(np.median(vals_cm))
        p95_height_cm = float(np.percentile(vals_cm, 95))
        p98_height_cm = float(np.percentile(vals_cm, 98))
        max_height_cm = float(np.max(vals_cm))
    area_batubara_cm2 = float(valid_pixels * area_per_pixel_cm2)

    massa_kg = None
    if AKTIFKAN_ESTIMASI_MASSA:
        massa_kg = volume_m3 * DENSITY_KG_PER_M3

    return {
        "area_per_pixel_m2": area_per_pixel_m2,
        "area_per_pixel_cm2": area_per_pixel_cm2,
        "pixel_length_m": pixel_length_m,
        "pixel_width_m": pixel_width_m,
        "valid_pixels": valid_pixels,
        "area_batubara_cm2": area_batubara_cm2,
        "volume_m3": volume_m3,
        "volume_cm3": volume_cm3,
        "volume_liter": volume_liter,
        "mean_height_cm": mean_height_cm,
        "median_height_cm": median_height_cm,
        "p95_height_cm": p95_height_cm,
        "p98_height_cm": p98_height_cm,
        "max_height_cm": max_height_cm,
        "massa_kg": massa_kg,
    }


# =========================================================
# VISUALISASI & SIMPAN
# =========================================================
DISPLAY_MIRROR_HORIZONTAL = True


def mirror_untuk_tampilan(img):
    return cv2.flip(img, 1) if DISPLAY_MIRROR_HORIZONTAL else img


def colorize_height(height_m, valid_mask):
    arr = height_m.copy().astype(np.float32)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if valid_mask.sum() > 0:
        max_val = max(float(np.percentile(arr[valid_mask], 95)), 1e-6)
        norm = np.clip(arr / max_val, 0, 1)
        out = (norm * 255).astype(np.uint8)
        out[~valid_mask] = 0
    color = cv2.applyColorMap(out, cv2.COLORMAP_TURBO)
    color[~valid_mask] = (0, 0, 0)
    return color


def overlay_mask(img_bgr, mask):
    color = np.zeros_like(img_bgr)
    color[mask] = (0, 255, 0)
    return cv2.addWeighted(img_bgr, 0.65, color, 0.35, 0)


def panel(img, title, w=400, h=160):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    header_h = 34
    body_h = h - header_h
    ih, iw = img.shape[:2]
    scale = min(w / iw, body_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    body = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x0 = (w - nw) // 2
    y0 = header_h + (body_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = body
    cv2.rectangle(canvas, (0, 0), (w, header_h), (35, 35, 35), -1)
    cv2.putText(canvas, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (80, 80, 80), 1)
    return canvas


def buat_dashboard(snapshot, rect_top, height_top, mask_top, valid_volume, stats):
    height_color = colorize_height(height_top, valid_volume)
    mask_vis = cv2.cvtColor((mask_top.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    overlay = overlay_mask(rect_top, mask_top)
    valid_vis = cv2.cvtColor((valid_volume.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)

    p1 = panel(mirror_untuk_tampilan(snapshot["rect_left"]), "RECTIFIED LEFT + ROI", 420, 250)
    p2 = panel(mirror_untuk_tampilan(rect_top), "TOP-VIEW GERBONG", 420, 250)
    p3 = panel(mirror_untuk_tampilan(overlay), "TOP-VIEW + MASK", 420, 250)
    p4 = panel(mirror_untuk_tampilan(mask_vis), "MASK TOP-VIEW", 420, 170)
    p5 = panel(mirror_untuk_tampilan(height_color), "HEIGHT TOP-VIEW", 420, 170)
    p6 = panel(mirror_untuk_tampilan(valid_vis), "VALID VOLUME PIXELS", 420, 170)

    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    footer = np.zeros((90, row1.shape[1], 3), dtype=np.uint8)
    cv2.rectangle(footer, (0, 0), (footer.shape[1], footer.shape[0]), (20, 20, 20), -1)

    line1 = (
        f"Volume = {stats['volume_cm3']:.2f} cm3 | {stats['volume_liter']:.4f} L | "
        f"{stats['volume_m3']:.8f} m3"
    )
    line2 = (
        f"Mean height = {stats['mean_height_cm']:.2f} cm | Median = {stats['median_height_cm']:.2f} cm | "
        f"Max = {stats['max_height_cm']:.2f} cm | Valid pixels = {stats['valid_pixels']}"
    )
    line3 = (
        f"Gerbong = {GERBONG_LENGTH_CM:.1f} cm x {GERBONG_WIDTH_CM:.1f} cm | "
        f"Top-view = {TOPVIEW_W} x {TOPVIEW_H} px | area/pixel = {stats['area_per_pixel_m2']:.10f} m2"
    )

    cv2.putText(footer, line1, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    cv2.putText(footer, line2, (12, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    cv2.putText(footer, line3, (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    return np.vstack([row1, row2, footer]), height_color, mask_vis, overlay, valid_vis


def gambar_roi_pada_rectified(rect_left, src_points):
    out = rect_left.copy()
    pts = src_points.astype(np.int32)
    cv2.polylines(out, [pts], True, (0, 255, 255), 3)
    for i, p in enumerate(pts):
        cv2.circle(out, tuple(p), 6, (0, 0, 255), -1)
        cv2.putText(out, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def simpan_hasil(snapshot, H, src_points, dst_points, rect_top, height_top, valid_top, mask_top, valid_volume, dashboard, height_color, mask_vis, overlay, valid_vis, stats):
    pastikan_folder_output()
    waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"volume_{snapshot['prefix']}_{waktu}"

    path_dashboard = os.path.join(OUTPUT_DIR, f"{prefix}_dashboard.png")
    path_rect_roi = os.path.join(OUTPUT_DIR, f"{prefix}_rectified_roi.png")
    path_top = os.path.join(OUTPUT_DIR, f"{prefix}_topview.png")
    path_mask = os.path.join(OUTPUT_DIR, f"{prefix}_mask_topview.png")
    path_height = os.path.join(OUTPUT_DIR, f"{prefix}_height_topview.png")
    path_overlay = os.path.join(OUTPUT_DIR, f"{prefix}_overlay_topview.png")
    path_valid = os.path.join(OUTPUT_DIR, f"{prefix}_valid_volume.png")
    path_npz = os.path.join(OUTPUT_DIR, f"{prefix}_volume_data.npz")
    path_txt = os.path.join(OUTPUT_DIR, f"{prefix}_ringkasan_volume.txt")
    path_csv = os.path.join(OUTPUT_DIR, "volume_history.csv")

    rect_roi = gambar_roi_pada_rectified(snapshot["rect_left"], src_points)

    cv2.imwrite(path_dashboard, dashboard)
    cv2.imwrite(path_rect_roi, rect_roi)
    cv2.imwrite(path_top, rect_top)
    cv2.imwrite(path_mask, (mask_top.astype(np.uint8) * 255))
    cv2.imwrite(path_height, height_color)
    cv2.imwrite(path_overlay, overlay)
    cv2.imwrite(path_valid, valid_vis)

    np.savez_compressed(
        path_npz,
        H=H,
        src_points=src_points,
        dst_points=dst_points,
        rect_top=rect_top,
        height_top_m=height_top.astype(np.float32),
        valid_top=valid_top.astype(np.uint8),
        mask_top=mask_top.astype(np.uint8),
        valid_volume=valid_volume.astype(np.uint8),
        area_per_pixel_m2=np.array([stats["area_per_pixel_m2"]], dtype=np.float32),
        area_per_pixel_cm2=np.array([stats["area_per_pixel_cm2"]], dtype=np.float32),
        area_batubara_cm2=np.array([stats["area_batubara_cm2"]], dtype=np.float32),
        volume_m3=np.array([stats["volume_m3"]], dtype=np.float32),
        volume_cm3=np.array([stats["volume_cm3"]], dtype=np.float32),
        volume_liter=np.array([stats["volume_liter"]], dtype=np.float32),
        mean_height_cm=np.array([stats["mean_height_cm"]], dtype=np.float32),
        median_height_cm=np.array([stats["median_height_cm"]], dtype=np.float32),
        p95_height_cm=np.array([stats["p95_height_cm"]], dtype=np.float32),
        p98_height_cm=np.array([stats["p98_height_cm"]], dtype=np.float32),
        max_height_cm=np.array([stats["max_height_cm"]], dtype=np.float32),
        floor_reference_mode=np.array([snapshot.get("floor_reference_mode", "unknown")]),
        floor_reference_path=np.array([snapshot.get("floor_reference_path", "")]),
    )

    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("RINGKASAN VOLUME BATUBARA BERBASIS HOMOGRAFI\n")
        f.write("================================================\n")
        f.write(f"Input NPZ              : {snapshot['npz_path']}\n")
        f.write(f"Input rectified left   : {snapshot['img_path']}\n")
        f.write(f"Sample ID              : {snapshot.get('sample_id', snapshot['prefix'])}\n")
        f.write(f"Baseline cm            : {snapshot.get('baseline_cm', '')}\n")
        f.write(f"Target weight gram     : {snapshot.get('target_weight_g', '')}\n")
        f.write(f"Actual weight gram     : diisi pada CSV kalibrasi massa\n")
        f.write(f"Condition              : {snapshot.get('condition', '')}\n")
        f.write(f"Profile                : {snapshot.get('profile', '')}\n")
        f.write(f"Repetition             : {snapshot.get('repetition', '')}\n")
        f.write(f"Floor reference mode   : {snapshot.get('floor_reference_mode', 'unknown')}\n")
        f.write(f"Floor reference path   : {snapshot.get('floor_reference_path', '')}\n")
        f.write(f"Gerbong length cm      : {GERBONG_LENGTH_CM:.3f}\n")
        f.write(f"Gerbong width cm       : {GERBONG_WIDTH_CM:.3f}\n")
        f.write(f"Gerbong height cm      : {GERBONG_HEIGHT_CM:.3f}\n")
        f.write(f"Top-view size px       : {TOPVIEW_W} x {TOPVIEW_H}\n")
        f.write(f"Area per pixel m2      : {stats['area_per_pixel_m2']:.12f}\n")
        f.write(f"Area per pixel cm2     : {stats['area_per_pixel_cm2']:.8f}\n")
        f.write(f"Valid pixels           : {stats['valid_pixels']}\n")
        f.write(f"Area batubara cm2      : {stats['area_batubara_cm2']:.6f}\n")
        f.write(f"Mean height cm         : {stats['mean_height_cm']:.6f}\n")
        f.write(f"Median height cm       : {stats['median_height_cm']:.6f}\n")
        f.write(f"P95 height cm          : {stats['p95_height_cm']:.6f}\n")
        f.write(f"P98 height cm          : {stats['p98_height_cm']:.6f}\n")
        f.write(f"Max height cm          : {stats['max_height_cm']:.6f}\n")
        f.write(f"Volume m3              : {stats['volume_m3']:.10f}\n")
        f.write(f"Volume cm3             : {stats['volume_cm3']:.6f}\n")
        f.write(f"Volume liter           : {stats['volume_liter']:.6f}\n")
        if stats["massa_kg"] is not None:
            f.write(f"Density kg/m3          : {DENSITY_KG_PER_M3:.3f}\n")
            f.write(f"Massa estimasi kg      : {stats['massa_kg']:.6f}\n")
        else:
            f.write("Massa estimasi kg      : belum dihitung; butuh massa jenis/kalibrasi empiris\n")
        f.write("\nCatatan: volume dihitung dari height_m hasil Script 11 terbaru. Script 12 tidak menghitung ulang tinggi dari Z_ref global.\n")

    csv_exists = os.path.exists(path_csv)
    with open(path_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow([
                "timestamp", "sample_id", "prefix", "baseline_cm", "target_weight_g",
                "actual_weight_g", "condition", "profile", "repetition", "notes",
                "snapshot_npz", "volume_npz", "floor_reference_mode", "floor_reference_path",
                "volume_m3", "volume_cm3", "volume_liter",
                "mean_height_cm", "median_height_cm", "p95_height_cm", "p98_height_cm",
                "max_height_cm", "valid_pixels", "area_batubara_cm2",
                "area_per_pixel_m2", "area_per_pixel_cm2", "massa_kg", "status"
            ])
        status = "valid" if stats["valid_pixels"] >= MIN_VALID_AREA_PIXELS and stats["volume_cm3"] > 0 else "perlu_cek"
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            snapshot.get("sample_id", snapshot["prefix"]), snapshot["prefix"],
            csv_value(snapshot.get("baseline_cm")), csv_value(snapshot.get("target_weight_g")),
            csv_value(snapshot.get("actual_weight_g")), snapshot.get("condition", ""),
            snapshot.get("profile", ""), csv_value(snapshot.get("repetition")),
            snapshot.get("notes", ""), snapshot.get("npz_path", ""), path_npz,
            snapshot.get("floor_reference_mode", "unknown"), snapshot.get("floor_reference_path", ""),
            stats["volume_m3"], stats["volume_cm3"], stats["volume_liter"],
            stats["mean_height_cm"], stats["median_height_cm"],
            stats["p95_height_cm"], stats["p98_height_cm"], stats["max_height_cm"],
            stats["valid_pixels"], stats["area_batubara_cm2"],
            stats["area_per_pixel_m2"], stats["area_per_pixel_cm2"],
            "" if stats["massa_kg"] is None else stats["massa_kg"], status,
        ])

    print("\n[OK] Hasil volume tersimpan")
    print(f"     Dashboard : {path_dashboard}")
    print(f"     Data NPZ  : {path_npz}")
    print(f"     Ringkasan : {path_txt}")
    print(f"     CSV       : {path_csv}")


# =========================================================
# MAIN
# =========================================================
def proses_volume_satu_snapshot(snapshot, src_points, tampilkan_dashboard=False):
    """Proses satu snapshot menjadi volume dan simpan hasil."""
    H, src_ordered, dst_points = buat_homografi(src_points)

    rect_top, height_top, valid_top, mask_top, valid_volume = warp_topview(
        snapshot["rect_left"],
        snapshot["height_m"],
        snapshot["valid_height"],
        snapshot["mask_unet"],
        H,
    )

    stats = hitung_volume(height_top, valid_volume)

    dashboard, height_color, mask_vis, overlay, valid_vis = buat_dashboard(
        snapshot, rect_top, height_top, mask_top, valid_volume, stats
    )

    print("\n" + "=" * 78)
    print("HASIL VOLUME")
    print("=" * 78)
    print(f"Snapshot          : {snapshot['prefix']}")
    print(f"Valid pixels      : {stats['valid_pixels']}")
    print(f"Mean height       : {stats['mean_height_cm']:.3f} cm")
    print(f"Median height     : {stats['median_height_cm']:.3f} cm")
    print(f"P95 height        : {stats['p95_height_cm']:.3f} cm")
    print(f"P98 height        : {stats['p98_height_cm']:.3f} cm")
    print(f"Max height        : {stats['max_height_cm']:.3f} cm")
    print(f"Area batubara     : {stats['area_batubara_cm2']:.3f} cm^2")
    print(f"Volume            : {stats['volume_cm3']:.3f} cm^3")
    print(f"Volume            : {stats['volume_liter']:.6f} liter")
    print(f"Volume            : {stats['volume_m3']:.10f} m^3")
    if stats["massa_kg"] is not None:
        print(f"Estimasi massa    : {stats['massa_kg']:.6f} kg")
    else:
        print("Estimasi massa    : belum dihitung; butuh massa jenis/kalibrasi empiris")
    print("=" * 78)

    simpan_hasil(
        snapshot, H, src_ordered, dst_points,
        rect_top, height_top, valid_top, mask_top, valid_volume,
        dashboard, height_color, mask_vis, overlay, valid_vis, stats,
    )

    if tampilkan_dashboard:
        dashboard_show, _ = resize_for_display(dashboard)
        cv2.namedWindow("Dashboard Volume Homografi", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dashboard Volume Homografi", dashboard_show.shape[1], dashboard_show.shape[0])
        cv2.imshow("Dashboard Volume Homografi", dashboard_show)
        print("\n[INFO] Tekan tombol apa saja pada jendela dashboard untuk keluar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return stats


def tulis_batch_summary_header(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "no", "snapshot_npz", "sample_id", "prefix", "baseline_cm", "target_weight_g",
            "actual_weight_g", "condition", "profile", "repetition",
            "volume_cm3", "volume_liter", "volume_m3",
            "mean_height_cm", "median_height_cm", "p95_height_cm", "p98_height_cm",
            "max_height_cm", "valid_pixels", "area_batubara_cm2",
            "area_per_pixel_m2", "area_per_pixel_cm2", "floor_reference_mode", "status"
        ])


def tulis_batch_summary_row(path, no, snapshot, stats):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        status = "valid" if stats["valid_pixels"] >= MIN_VALID_AREA_PIXELS and stats["volume_cm3"] > 0 else "perlu_cek"
        writer.writerow([
            no, snapshot["npz_path"], snapshot.get("sample_id", snapshot["prefix"]), snapshot["prefix"],
            csv_value(snapshot.get("baseline_cm")), csv_value(snapshot.get("target_weight_g")),
            csv_value(snapshot.get("actual_weight_g")), snapshot.get("condition", ""),
            snapshot.get("profile", ""), csv_value(snapshot.get("repetition")),
            stats["volume_cm3"], stats["volume_liter"], stats["volume_m3"],
            stats["mean_height_cm"], stats["median_height_cm"],
            stats["p95_height_cm"], stats["p98_height_cm"], stats["max_height_cm"],
            stats["valid_pixels"], stats["area_batubara_cm2"],
            stats["area_per_pixel_m2"], stats["area_per_pixel_cm2"],
            snapshot.get("floor_reference_mode", "unknown"), status,
        ])


def pilih_roi_untuk_mode(snapshots_npz):
    """Deteksi ROI otomatis; jika belum yakin, gunakan ROI lama atau klik manual."""
    first_snapshot = load_snapshot(snapshots_npz[0])

    if AUTO_DETECT_ROI:
        roi_auto, score, info, debug_path = deteksi_roi_gerbong_otomatis(first_snapshot["rect_left"])
        if roi_auto is not None:
            print("\n[INFO] Deteksi ROI otomatis selesai.")
            print(f"       Score      : {score:.3f}")
            print(f"       Aspect     : {info.get('aspect', 0.0):.3f} | target {(GERBONG_LENGTH_CM / GERBONG_WIDTH_CM):.3f}")
            print(f"       Area ratio : {info.get('area_ratio', 0.0):.3f}")
            if debug_path:
                print(f"       Debug ROI  : {debug_path}")

            if score >= AUTO_ROI_MIN_CONFIDENCE:
                if AUTO_ROI_CONFIRM_BEFORE_USE:
                    ans = input("Gunakan ROI otomatis ini? [Y/n]: ").strip().lower()
                    if ans not in ["", "y", "yes", "ya"]:
                        print("[INFO] ROI otomatis ditolak. Lanjut fallback.")
                    else:
                        simpan_roi_points(roi_auto)
                        return roi_auto
                else:
                    print("[OK] ROI otomatis dipakai karena confidence memenuhi batas.")
                    simpan_roi_points(roi_auto)
                    return roi_auto
            else:
                print(
                    f"[WARNING] ROI otomatis belum cukup yakin "
                    f"({score:.3f} < {AUTO_ROI_MIN_CONFIDENCE:.3f}). Lanjut fallback."
                )
        else:
            print("[WARNING] ROI otomatis tidak menemukan kandidat yang layak. Lanjut fallback.")

    roi = load_roi_points()
    if roi is not None:
        print("\nGunakan ROI lama untuk semua data? [Y/n]")
        ans = input("Jawaban: ").strip().lower()
        if ans in ["", "y", "yes", "ya"]:
            return roi

    print("\n[INFO] Pilih ROI sekali. ROI ini akan dipakai untuk semua snapshot.")
    roi = pilih_roi_interaktif(first_snapshot["rect_left"])
    return roi


def main_single():
    snapshot = load_snapshot()
    src_points = pilih_roi_untuk_mode([snapshot["npz_path"]])
    proses_volume_satu_snapshot(snapshot, src_points, tampilkan_dashboard=True)


def main_batch():
    npz_files = cari_semua_npz()
    print(f"\n[INFO] Mode batch aktif. Ditemukan {len(npz_files)} snapshot dari Script 11:")
    for i, p in enumerate(npz_files, start=1):
        print(f"{i:03d}. {os.path.basename(p)}")

    jawab = input("\nProses semua snapshot di atas? [Y/n]: ").strip().lower()
    if jawab not in ["", "y", "yes", "ya"]:
        print("[BATAL] Batch dibatalkan.")
        return

    src_points = pilih_roi_untuk_mode(npz_files)

    waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_csv = os.path.join(OUTPUT_DIR, f"batch_volume_summary_{waktu}.csv")
    tulis_batch_summary_header(batch_csv)

    # Bangun ulang volume_history.csv agar dataset kalibrasi tidak duplikat.
    volume_history_path = os.path.join(OUTPUT_DIR, "volume_history.csv")
    if OVERWRITE_VOLUME_HISTORY_ON_BATCH and os.path.exists(volume_history_path):
        os.remove(volume_history_path)
        print(f"[INFO] volume_history lama dihapus agar batch bersih: {volume_history_path}")

    for i, npz_path in enumerate(npz_files, start=1):
        print("\n" + "#" * 78)
        print(f"BATCH {i}/{len(npz_files)}")
        print("#" * 78)
        snapshot = load_snapshot(npz_path)
        stats = proses_volume_satu_snapshot(snapshot, src_points, tampilkan_dashboard=False)
        tulis_batch_summary_row(batch_csv, i, snapshot, stats)

    print("\n" + "=" * 78)
    print("BATCH SELESAI")
    print("=" * 78)
    print(f"Jumlah snapshot diproses : {len(npz_files)}")
    print(f"Batch summary CSV        : {batch_csv}")
    print(f"Volume history utama     : {os.path.join(OUTPUT_DIR, 'volume_history.csv')}")
    print("\nGunakan volume_history.csv untuk Script 13B.")


def main():
    print("\n" + "=" * 78)
    print("SCRIPT 12 — HITUNG VOLUME HOMOGRAFI + METADATA KALIBRASI MASSA")
    print("VERSI GABUNGAN SINGLE + BATCH")
    print("=" * 78)
    print(f"Mode             : {MODE}")
    print(f"Input dir        : {INPUT_DIR}")
    print(f"Output dir       : {OUTPUT_DIR}")
    print(f"Gerbong          : {GERBONG_LENGTH_CM:.1f} cm x {GERBONG_WIDTH_CM:.1f} cm x {GERBONG_HEIGHT_CM:.1f} cm")
    print(f"Top-view         : {TOPVIEW_W} x {TOPVIEW_H} px")
    print(f"Height max filter: {HEIGHT_MAX_M*100:.2f} cm")
    print("=" * 78)

    pastikan_folder_output()

    if MODE.lower() == "single":
        main_single()
    elif MODE.lower() == "batch":
        main_batch()
    else:
        raise ValueError("MODE harus 'single' atau 'batch'.")


if __name__ == "__main__":
    main()
