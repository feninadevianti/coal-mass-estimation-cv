"""
SCRIPT 4 - REKTIFIKASI + PEMETAAN STEREO + REAL-TIME REMAP
==========================================================

Fungsi utama:
1. Memuat parameter hasil kalibrasi ekstrinsik/stereo dari XML.
2. Jika stereomap.xml belum lengkap, script dapat membuat ulang:
   - R1, R2, P1, P2, Q dengan cv2.stereoRectify()
   - stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y dengan cv2.initUndistortRectifyMap()
   - menyimpan semuanya ke hasil_kalibrasi/stereomap.xml memakai cv2.FileStorage
3. Memuat stereomap.xml untuk aplikasi real-time.
4. Menerapkan cv2.remap() pada frame kamera kiri dan kanan.
5. Menampilkan hasil rektifikasi secara real-time.
6. Menyimpan screenshot hasil remap jika tombol S ditekan.

Script ini diselaraskan dengan:
- Script 2: kalibrasi intrinsik ChArUco XML
  hasil_kalibrasi/intrinsik_kiri.xml
  hasil_kalibrasi/intrinsik_kanan.xml

- Script 3: kalibrasi ekstrinsik/stereo ChArUco XML
  hasil_kalibrasi/stereomap.xml
  hasil_kalibrasi/stereo_params.xml

Cara pakai:
python 04_rektifikasi_stereomap_realtime_SELARAS_06_CROP.py

Kontrol saat jendela tampil:
[S]   Simpan screenshot
[G]   Tampilkan/sembunyikan garis epipolar horizontal
[ESC] Keluar
"""

import cv2
import numpy as np
import os
from datetime import datetime


# =========================================================
# KONFIGURASI
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "hasil_kalibrasi")

INTRINSIK_KIRI_XML = os.path.join(OUTPUT_DIR, "intrinsik_kiri.xml")
INTRINSIK_KANAN_XML = os.path.join(OUTPUT_DIR, "intrinsik_kanan.xml")
STEREO_PARAMS_XML = os.path.join(OUTPUT_DIR, "stereo_params.xml")
STEREOMAP_XML = os.path.join(OUTPUT_DIR, "stereomap.xml")

REALTIME_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "realtime_remap")

CAMERA_ID_KIRI = 0
CAMERA_ID_KANAN = 1

# Sesuaikan dengan resolusi saat kalibrasi. Jika XML menyimpan image_width dan image_height,
# nilai dari XML akan dipakai sebagai acuan utama.
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

# Jika kamera tidak berhasil memberi resolusi sesuai kalibrasi, frame dapat di-resize untuk uji tampilan.
# Untuk pengukuran final, sebaiknya False dan pastikan kamera benar-benar memakai resolusi kalibrasi.
ALLOW_RESIZE_IF_NEEDED = True

# Jika True, script akan menghitung ulang R1, R2, P1, P2, Q, dan stereo map
# agar output matriks rektifikasi selalu tampil di terminal.
FORCE_REGENERATE_STEREOMAP = True

# Output tambahan
MATRIX_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "matrix_output")
BEFORE_AFTER_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "before_after_rectification")

# Stereo map berukuran sangat besar. False = terminal hanya menampilkan ringkasan dan sampel.
# File TXT dan gambar tetap disimpan.
PRINT_FULL_STEREO_MAP = False
STEREO_MAP_SAMPLE_SIZE = 6

# Simpan otomatis 1 gambar sebelum/sesudah rektifikasi saat frame pertama berhasil dibaca.
AUTO_SAVE_FIRST_FRAME = True


# =========================================================
# KONFIGURASI SELARAS SCRIPT 06 CAPTURE / CROP PIPELINE
# =========================================================
# Script 06 menyimpan gambar kalibrasi dalam kondisi crop-resize.
# Artinya frame live dari kamera juga harus di-crop dengan ROI yang sama sebelum cv2.remap.
CROP_AKTIF = True
SIMPAN_FRAME_CROP = True
TERAPKAN_CROP_REALTIME = True

ROI_LEFT = (0.00, 0.05, 1.00, 0.95)
ROI_RIGHT = (0.00, 0.05, 1.00, 0.95)

# Ukuran raw dari kamera sebelum crop. Pada setup kamu biasanya 640x480.
CAMERA_RAW_WIDTH = 640
CAMERA_RAW_HEIGHT = 480
FPS = 30

# Setting kamera diselaraskan dengan Script 06 Capture ChArUco.
SAMAKAN_SETTING_KAMERA = True
WB_TEMPERATURE = 4500
BRIGHTNESS_VALUE = 128
CONTRAST_VALUE = 128
GAIN_VALUE = 0
GUNAKAN_MANUAL_EXPOSURE = False
EXPOSURE_VALUE = -6
KUNCI_AUTOFOCUS = False
FOCUS_LEFT = 0
FOCUS_RIGHT = 0

CAMERA_HEIGHT_CM = 60.0
CAMERA_TILT_MIN_DEG = 10.0
CAMERA_TILT_MAX_DEG = 15.0
CAMERA_FOV_DEG = 120.0

GERBONG_PANJANG_CM = 40.0
GERBONG_LEBAR_CM = 10.0
GERBONG_TINGGI_CM = 7.5

# Peringatan kualitas stereo. Jika RMS stereo lebih tinggi dari batas ini,
# script tetap bisa berjalan untuk cek visual, tetapi jangan dipakai untuk pengukuran final.
RMS_STEREO_BATAS_AMAN = 1.0

# Pengaturan tampilan agar seluruh before/after terlihat di layar laptop.
WINDOW_NAME_REALTIME = "Output Realtime: Sebelum vs Setelah Rektifikasi [S=Simpan | G=Garis | ESC=Keluar]"
DISPLAY_MAX_WIDTH = 1180
DISPLAY_MAX_HEIGHT = 680
DISPLAY_WINDOW_NORMAL = True
DISPLAY_MIRROR_HORIZONTAL = True


# =========================================================
# UTILITAS XML OPENCV
# =========================================================
def _node_empty(node):
    try:
        return node.empty()
    except Exception:
        return True


def buka_xml_read(path_xml):
    if not os.path.exists(path_xml):
        raise FileNotFoundError(f"File tidak ditemukan: {path_xml}")
    fs = cv2.FileStorage(path_xml, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Gagal membuka XML: {path_xml}")
    return fs


def baca_matrix(fs, nama_node, wajib=True):
    node = fs.getNode(nama_node)
    if _node_empty(node):
        if wajib:
            raise RuntimeError(f"Node '{nama_node}' tidak ditemukan di XML.")
        return None
    mat = node.mat()
    if mat is None and wajib:
        raise RuntimeError(f"Node '{nama_node}' tidak bisa dibaca sebagai matrix.")
    return mat


def baca_float(fs, nama_node, default=None):
    node = fs.getNode(nama_node)
    if _node_empty(node):
        return default
    try:
        return float(node.real())
    except Exception:
        return default


def baca_int(fs, nama_node, default=None):
    val = baca_float(fs, nama_node, default=None)
    if val is None:
        return default
    return int(round(val))


def baca_string(fs, nama_node, default=""):
    node = fs.getNode(nama_node)
    if _node_empty(node):
        return default
    try:
        s = node.string()
        return s if s is not None else default
    except Exception:
        return default


def cek_node_ada(path_xml, nama_node):
    if not os.path.exists(path_xml):
        return False
    fs = cv2.FileStorage(path_xml, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return False
    node = fs.getNode(nama_node)
    ada = not _node_empty(node)
    fs.release()
    return ada


# =========================================================
# ROI / CROP REALTIME - SELARAS SCRIPT 06
# =========================================================
def clamp_roi(roi):
    x1, y1, x2, y2 = roi
    x1 = max(0.0, min(0.95, float(x1)))
    y1 = max(0.0, min(0.95, float(y1)))
    x2 = max(0.05, min(1.0, float(x2)))
    y2 = max(0.05, min(1.0, float(y2)))

    if x2 <= x1:
        x1, x2 = 0.0, 1.0
    if y2 <= y1:
        y1, y2 = 0.0, 1.0

    return x1, y1, x2, y2


def crop_resize_frame(frame, roi, output_size):
    if frame is None:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_roi(roi)

    px1 = int(round(x1 * w))
    py1 = int(round(y1 * h))
    px2 = int(round(x2 * w))
    py2 = int(round(y2 * h))

    px1 = max(0, min(w - 2, px1))
    py1 = max(0, min(h - 2, py1))
    px2 = max(px1 + 2, min(w, px2))
    py2 = max(py1 + 2, min(h, py2))

    crop = frame[py1:py2, px1:px2]
    crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)
    return crop


def pilih_roi(nama_kamera):
    nama = str(nama_kamera).lower()
    if "kiri" in nama or nama == "left":
        return ROI_LEFT
    return ROI_RIGHT


def persiapkan_frame_realtime(frame, nama_kamera, image_size):
    """Menyamakan frame live dengan format gambar saat kalibrasi.

    Jika kalibrasi memakai crop-resize dari Script 06, maka frame live juga wajib
    di-crop dengan ROI yang sama sebelum masuk cv2.remap.
    """
    if frame is None:
        return None

    if TERAPKAN_CROP_REALTIME:
        return crop_resize_frame(frame, pilih_roi(nama_kamera), image_size)

    return frame


# =========================================================
# OUTPUT MATRIX: TERMINAL + TXT + GAMBAR PNG
# =========================================================
def nama_file_aman(teks):
    aman = "".join(c if c.isalnum() else "_" for c in teks)
    while "__" in aman:
        aman = aman.replace("__", "_")
    return aman.strip("_")


def matrix_to_string(mat, precision=8, max_rows=None, max_cols=None):
    if mat is None:
        return "Tidak tersedia"

    arr = np.asarray(mat)

    if arr.ndim == 0:
        return str(arr)

    # Untuk stereo map besar, tampilkan sampel agar terminal tetap terbaca.
    if max_rows is not None and max_cols is not None and arr.ndim >= 2:
        arr_show = arr[:max_rows, :max_cols]
        teks = np.array2string(arr_show, precision=precision, suppress_small=False)
        teks += f"\n\n[Ringkasan] shape={arr.shape}, dtype={arr.dtype}"
        if arr.size > 0:
            try:
                teks += f", min={np.nanmin(arr):.6f}, max={np.nanmax(arr):.6f}, mean={np.nanmean(arr):.6f}"
            except Exception:
                pass
        teks += "\nCatatan: stereo map penuh disimpan ke file TXT/gambar, terminal hanya menampilkan sampel."
        return teks

    return np.array2string(arr, precision=precision, suppress_small=False)


def print_matrix_terminal(nama, mat, full=True):
    print("\n" + "=" * 90)
    print(f"[{nama}]")
    print("=" * 90)

    if mat is None:
        print("Tidak tersedia")
        return

    arr = np.asarray(mat)
    print(f"Shape : {arr.shape}")
    print(f"Dtype : {arr.dtype}")

    if (not full) and arr.ndim >= 2:
        print(matrix_to_string(arr, max_rows=STEREO_MAP_SAMPLE_SIZE, max_cols=STEREO_MAP_SAMPLE_SIZE))
    else:
        print(matrix_to_string(arr))


def simpan_matrix_txt(path_txt, judul, mat, full=True):
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(judul + "\n")
        f.write("=" * len(judul) + "\n\n")
        if mat is None:
            f.write("Tidak tersedia\n")
            return

        arr = np.asarray(mat)
        f.write(f"Shape : {arr.shape}\n")
        f.write(f"Dtype : {arr.dtype}\n\n")

        if (not full) and arr.ndim >= 2:
            f.write(matrix_to_string(arr, max_rows=STEREO_MAP_SAMPLE_SIZE, max_cols=STEREO_MAP_SAMPLE_SIZE))
        else:
            f.write(matrix_to_string(arr))


def simpan_teks_ke_gambar(path_png, judul, teks, lebar=1800, tinggi=None):
    lines = []
    for line in teks.splitlines():
        if len(line) <= 170:
            lines.append(line)
        else:
            for i in range(0, len(line), 170):
                lines.append(line[i:i+170])

    if tinggi is None:
        tinggi = max(500, 70 + 24 * (len(lines) + 3))

    img = np.ones((tinggi, lebar, 3), dtype=np.uint8) * 255

    y = 40
    cv2.putText(img, judul[:100], (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    y += 35
    cv2.line(img, (25, y), (lebar - 25, y), (0, 0, 0), 1)
    y += 30

    for line in lines:
        if y > tinggi - 30:
            break
        cv2.putText(img, line[:170], (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
        y += 24

    cv2.imwrite(path_png, img)


def simpan_matrix_visual(path_png, judul, mat):
    if mat is None:
        simpan_teks_ke_gambar(path_png, judul, "Tidak tersedia")
        return

    arr = np.asarray(mat, dtype=np.float32)

    # Jika matrix 1D, ubah menjadi 2D.
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Untuk stereo map besar, downsample agar gambar tidak terlalu massa.
    max_dim = 800
    h, w = arr.shape[:2]
    if h > max_dim or w > max_dim:
        scale = min(max_dim / h, max_dim / w)
        arr_vis = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))))
    else:
        arr_vis = arr.copy()

    if arr_vis.ndim > 2:
        arr_vis = arr_vis[:, :, 0]

    # Normalisasi untuk heatmap.
    arr_norm = arr_vis.copy()
    arr_norm[~np.isfinite(arr_norm)] = 0

    if arr_norm.size == 0 or np.nanmax(arr_norm) == np.nanmin(arr_norm):
        gray = np.zeros(arr_norm.shape, dtype=np.uint8)
    else:
        gray = cv2.normalize(arr_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    heat = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    header = np.ones((70, heat.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(header, judul[:80], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.putText(header, f"shape={np.asarray(mat).shape}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    out = np.vstack([header, heat])
    cv2.imwrite(path_png, out)


def tampilkan_dan_simpan_semua_matriks(matrix_dict, output_dir=MATRIX_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    laporan_txt = os.path.join(output_dir, "semua_output_matriks.txt")
    laporan_lines = []

    for nama, mat in matrix_dict.items():
        is_stereo_map = nama.startswith("stereoMap")
        full_terminal = PRINT_FULL_STEREO_MAP if is_stereo_map else True

        print_matrix_terminal(nama, mat, full=full_terminal)

        safe = nama_file_aman(nama)
        path_txt = os.path.join(output_dir, f"{safe}.txt")
        path_png_text = os.path.join(output_dir, f"{safe}_teks.png")
        path_png_visual = os.path.join(output_dir, f"{safe}_visual.png")

        # TXT penuh untuk matrix kecil, sampel untuk stereo map agar file tetap masuk akal.
        simpan_matrix_txt(path_txt, nama, mat, full=full_terminal)

        teks_gambar = matrix_to_string(
            mat,
            max_rows=STEREO_MAP_SAMPLE_SIZE if is_stereo_map else None,
            max_cols=STEREO_MAP_SAMPLE_SIZE if is_stereo_map else None
        )
        simpan_teks_ke_gambar(path_png_text, nama, teks_gambar)
        simpan_matrix_visual(path_png_visual, nama, mat)

        laporan_lines.append("\n" + "=" * 90)
        laporan_lines.append(nama)
        laporan_lines.append("=" * 90)
        laporan_lines.append(matrix_to_string(
            mat,
            max_rows=STEREO_MAP_SAMPLE_SIZE if is_stereo_map else None,
            max_cols=STEREO_MAP_SAMPLE_SIZE if is_stereo_map else None
        ))

    with open(laporan_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(laporan_lines))

    simpan_teks_ke_gambar(
        os.path.join(output_dir, "dashboard_ringkasan_matriks.png"),
        "Dashboard Ringkasan Output Matriks",
        "\n".join(laporan_lines),
        lebar=2000
    )

    print("\n" + "=" * 90)
    print("[OK] Semua output matriks tersimpan")
    print(f"Folder: {output_dir}")
    print("=" * 90)


def buat_matrix_dict_dari_data(
    intrinsik_kiri=None,
    intrinsik_kanan=None,
    stereo_params=None,
    R1=None,
    R2=None,
    P1=None,
    P2=None,
    Q=None,
    roi1=None,
    roi2=None,
    mapLx=None,
    mapLy=None,
    mapRx=None,
    mapRy=None,
    stereomap=None
):
    data = {}

    if intrinsik_kiri is not None:
        data["cameraMatrixL_intrinsik_kiri"] = intrinsik_kiri.get("camera_matrix")
        data["distCoeffsL_intrinsik_kiri"] = intrinsik_kiri.get("dist_coeffs")

    if intrinsik_kanan is not None:
        data["cameraMatrixR_intrinsik_kanan"] = intrinsik_kanan.get("camera_matrix")
        data["distCoeffsR_intrinsik_kanan"] = intrinsik_kanan.get("dist_coeffs")

    if stereo_params is not None:
        data["R_rotation_matrix_ekstrinsik"] = stereo_params.get("R")
        data["T_translation_vector_ekstrinsik"] = stereo_params.get("T")
        data["E_essential_matrix"] = stereo_params.get("E")
        data["F_fundamental_matrix"] = stereo_params.get("F")
        data["baseline_m"] = np.array([[stereo_params.get("baseline_m")]], dtype=np.float64)
        data["baseline_mm"] = np.array([[stereo_params.get("baseline_mm")]], dtype=np.float64)
        data["rms_stereo"] = np.array([[stereo_params.get("rms_stereo")]], dtype=np.float64)
        data["valid_pairs"] = np.array([[stereo_params.get("valid_pairs")]], dtype=np.float64)

    if R1 is not None:
        data["R1_rectification_matrix_left"] = R1
        data["R2_rectification_matrix_right"] = R2
        data["P1_projection_matrix_left"] = P1
        data["P2_projection_matrix_right"] = P2
        data["Q_disparity_to_depth_matrix"] = Q
        data["roi1_left"] = np.array(roi1, dtype=np.int32)
        data["roi2_right"] = np.array(roi2, dtype=np.int32)

    if mapLx is not None:
        data["stereoMapL_x"] = mapLx
        data["stereoMapL_y"] = mapLy
        data["stereoMapR_x"] = mapRx
        data["stereoMapR_y"] = mapRy

    if stereomap is not None:
        for key in [
            "cameraMatrixL", "distCoeffsL", "cameraMatrixR", "distCoeffsR",
            "R", "T", "E", "F", "R1", "R2", "P1", "P2", "Q",
            "roi1", "roi2", "baseline_m", "baseline_mm", "rms_stereo", "valid_pairs",
            "stereoMapL_x", "stereoMapL_y", "stereoMapR_x", "stereoMapR_y"
        ]:
            if key in stereomap and stereomap[key] is not None:
                data[key] = stereomap[key]

    return data


def simpan_gambar_sebelum_sesudah(frameL, frameR, rectL, rectR, gabung_display, counter):
    os.makedirs(BEFORE_AFTER_OUTPUT_DIR, exist_ok=True)

    waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{counter:03d}_{waktu}"

    sebelum = np.hstack([
        tambah_label(frameL.copy(), "KIRI - SEBELUM REKTIFIKASI"),
        tambah_label(frameR.copy(), "KANAN - SEBELUM REKTIFIKASI")
    ])

    sesudah = np.hstack([
        tambah_label(rectL.copy(), "KIRI - SETELAH REKTIFIKASI"),
        tambah_label(rectR.copy(), "KANAN - SETELAH REKTIFIKASI")
    ])

    path_sebelum = os.path.join(BEFORE_AFTER_OUTPUT_DIR, f"{prefix}_sebelum_rektifikasi.png")
    path_sesudah = os.path.join(BEFORE_AFTER_OUTPUT_DIR, f"{prefix}_setelah_rektifikasi.png")
    path_realtime = os.path.join(BEFORE_AFTER_OUTPUT_DIR, f"{prefix}_output_realtime_before_after.png")

    cv2.imwrite(path_sebelum, sebelum)
    cv2.imwrite(path_sesudah, sesudah)
    cv2.imwrite(path_realtime, gabung_display)

    print(f"[OK] Gambar sebelum rektifikasi tersimpan : {path_sebelum}")
    print(f"[OK] Gambar setelah rektifikasi tersimpan : {path_sesudah}")
    print(f"[OK] Output realtime tersimpan            : {path_realtime}")



# =========================================================
# LOAD INTRINSIK DAN STEREO PARAMS
# =========================================================
def muat_intrinsik(path_xml, nama_kamera):
    fs = buka_xml_read(path_xml)
    camera_matrix = baca_matrix(fs, "camera_matrix")
    dist_coeffs = baca_matrix(fs, "dist_coeffs")

    image_width = baca_int(fs, "image_width", default=None)
    image_height = baca_int(fs, "image_height", default=None)

    if image_width is None or image_height is None:
        image_size_mat = baca_matrix(fs, "image_size", wajib=False)
        if image_size_mat is not None:
            flat = np.asarray(image_size_mat).astype(int).ravel()
            image_width = int(flat[0])
            image_height = int(flat[1])
        else:
            image_width = DEFAULT_WIDTH
            image_height = DEFAULT_HEIGHT

    mean_error = baca_float(fs, "mean_error", default=-1.0)
    fs.release()

    print(f"[OK] Intrinsik {nama_kamera} dimuat: {path_xml}")
    print(f"     image size : {(image_width, image_height)}")
    if mean_error >= 0:
        print(f"     mean error : {mean_error:.6f} px")

    return {
        "camera_matrix": camera_matrix.astype(np.float64),
        "dist_coeffs": dist_coeffs.astype(np.float64),
        "image_size": (image_width, image_height),
        "mean_error": mean_error,
    }


def muat_stereo_params(path_xml):
    fs = buka_xml_read(path_xml)
    R = baca_matrix(fs, "R")
    T = baca_matrix(fs, "T")
    E = baca_matrix(fs, "E", wajib=False)
    F = baca_matrix(fs, "F", wajib=False)
    baseline_m = baca_float(fs, "baseline_m", default=None)
    baseline_mm = baca_float(fs, "baseline_mm", default=None)
    rms_stereo = baca_float(fs, "rms_stereo", default=None)
    image_width = baca_int(fs, "image_width", default=DEFAULT_WIDTH)
    image_height = baca_int(fs, "image_height", default=DEFAULT_HEIGHT)
    valid_pairs = baca_int(fs, "valid_pairs", default=-1)
    fs.release()

    if baseline_m is None:
        baseline_m = float(np.linalg.norm(T))
    if baseline_mm is None:
        baseline_mm = baseline_m * 1000.0

    print(f"[OK] Stereo params dimuat: {path_xml}")
    print(f"     RMS stereo  : {rms_stereo:.6f} px" if rms_stereo is not None else "     RMS stereo  : tidak tersedia")
    if rms_stereo is not None and rms_stereo > RMS_STEREO_BATAS_AMAN:
        print(f"     [PERINGATAN] RMS stereo > {RMS_STEREO_BATAS_AMAN} px. Gunakan hanya untuk cek visual, belum untuk pengukuran final.")
    print(f"     Baseline    : {baseline_m:.6f} m | {baseline_mm:.3f} mm")
    print(f"     Valid pairs : {valid_pairs}" if valid_pairs >= 0 else "     Valid pairs : tidak tersedia")

    return {
        "R": R.astype(np.float64),
        "T": T.astype(np.float64),
        "E": None if E is None else E.astype(np.float64),
        "F": None if F is None else F.astype(np.float64),
        "baseline_m": baseline_m,
        "baseline_mm": baseline_mm,
        "rms_stereo": rms_stereo,
        "image_size": (image_width, image_height),
        "valid_pairs": valid_pairs,
    }


def muat_stereomap(path_xml):
    fs = buka_xml_read(path_xml)
    mapLx = baca_matrix(fs, "stereoMapL_x")
    mapLy = baca_matrix(fs, "stereoMapL_y")
    mapRx = baca_matrix(fs, "stereoMapR_x")
    mapRy = baca_matrix(fs, "stereoMapR_y")

    Q = baca_matrix(fs, "Q", wajib=False)
    R = baca_matrix(fs, "R", wajib=False)
    T = baca_matrix(fs, "T", wajib=False)
    E = baca_matrix(fs, "E", wajib=False)
    F = baca_matrix(fs, "F", wajib=False)

    R1 = baca_matrix(fs, "R1", wajib=False)
    R2 = baca_matrix(fs, "R2", wajib=False)
    P1 = baca_matrix(fs, "P1", wajib=False)
    P2 = baca_matrix(fs, "P2", wajib=False)

    cameraMatrixL = baca_matrix(fs, "cameraMatrixL", wajib=False)
    distCoeffsL = baca_matrix(fs, "distCoeffsL", wajib=False)
    cameraMatrixR = baca_matrix(fs, "cameraMatrixR", wajib=False)
    distCoeffsR = baca_matrix(fs, "distCoeffsR", wajib=False)

    roi1 = baca_matrix(fs, "roi1", wajib=False)
    roi2 = baca_matrix(fs, "roi2", wajib=False)

    baseline_m = baca_float(fs, "baseline_m", default=None)
    baseline_mm = baca_float(fs, "baseline_mm", default=None)
    rms_stereo = baca_float(fs, "rms_stereo", default=None)
    valid_pairs = baca_int(fs, "valid_pairs", default=None)

    image_width = baca_int(fs, "image_width", default=mapLx.shape[1])
    image_height = baca_int(fs, "image_height", default=mapLx.shape[0])
    map_type = baca_string(fs, "map_type", default="tidak tersedia")
    fs.release()

    print(f"[OK] Stereomap dimuat: {path_xml}")
    print(f"     map size : {(image_width, image_height)}")
    print(f"     map type : {map_type}")

    return {
        "stereoMapL_x": mapLx,
        "stereoMapL_y": mapLy,
        "stereoMapR_x": mapRx,
        "stereoMapR_y": mapRy,
        "Q": Q,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "cameraMatrixL": cameraMatrixL,
        "distCoeffsL": distCoeffsL,
        "cameraMatrixR": cameraMatrixR,
        "distCoeffsR": distCoeffsR,
        "roi1": roi1,
        "roi2": roi2,
        "baseline_m": np.array([[baseline_m]], dtype=np.float64) if baseline_m is not None else None,
        "baseline_mm": np.array([[baseline_mm]], dtype=np.float64) if baseline_mm is not None else None,
        "rms_stereo": np.array([[rms_stereo]], dtype=np.float64) if rms_stereo is not None else None,
        "valid_pairs": np.array([[valid_pairs]], dtype=np.float64) if valid_pairs is not None else None,
        "image_size": (image_width, image_height),
        "map_type": map_type,
    }



# =========================================================
# REKTIFIKASI, PEMETAAN STEREO, DAN PENYIMPANAN XML
# =========================================================
def simpan_stereomap_xml(
    path_xml,
    mapLx,
    mapLy,
    mapRx,
    mapRy,
    Q,
    R,
    T,
    E,
    F,
    R1,
    R2,
    P1,
    P2,
    roi1,
    roi2,
    intrinsik_kiri,
    intrinsik_kanan,
    stereo_params,
    image_size
):
    fs = cv2.FileStorage(path_xml, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Gagal membuat XML: {path_xml}")

    fs.write("stereoMapL_x", mapLx)
    fs.write("stereoMapL_y", mapLy)
    fs.write("stereoMapR_x", mapRx)
    fs.write("stereoMapR_y", mapRy)

    fs.write("R", R)
    fs.write("T", T)
    if E is not None:
        fs.write("E", E)
    if F is not None:
        fs.write("F", F)

    fs.write("R1", R1)
    fs.write("R2", R2)
    fs.write("P1", P1)
    fs.write("P2", P2)
    fs.write("Q", Q)

    fs.write("cameraMatrixL", intrinsik_kiri["camera_matrix"])
    fs.write("distCoeffsL", intrinsik_kiri["dist_coeffs"])
    fs.write("cameraMatrixR", intrinsik_kanan["camera_matrix"])
    fs.write("distCoeffsR", intrinsik_kanan["dist_coeffs"])

    fs.write("roi1", np.array(roi1, dtype=np.int32))
    fs.write("roi2", np.array(roi2, dtype=np.int32))
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("map_type", "CV_32FC1")
    fs.write("map_description", "stereoMap*_x dan stereoMap*_y adalah peta remap arah x dan y float32 untuk cv2.remap")
    fs.write("capture_script_reference", "06_CAPTURE_CHARUCO_STEREO_LEFT_BOOST_MULTI_DETECT")
    fs.write("crop_aktif", int(CROP_AKTIF))
    fs.write("simpan_frame_crop", int(SIMPAN_FRAME_CROP))
    fs.write("terapkan_crop_realtime", int(TERAPKAN_CROP_REALTIME))
    fs.write("roi_left_capture", np.array(ROI_LEFT, dtype=np.float32).reshape(1, 4))
    fs.write("roi_right_capture", np.array(ROI_RIGHT, dtype=np.float32).reshape(1, 4))
    fs.write("camera_raw_width", int(CAMERA_RAW_WIDTH))
    fs.write("camera_raw_height", int(CAMERA_RAW_HEIGHT))
    fs.write("camera_height_cm", float(CAMERA_HEIGHT_CM))
    fs.write("camera_tilt_min_deg", float(CAMERA_TILT_MIN_DEG))
    fs.write("camera_tilt_max_deg", float(CAMERA_TILT_MAX_DEG))
    fs.write("camera_fov_deg", float(CAMERA_FOV_DEG))
    fs.write("gerbong_panjang_cm", float(GERBONG_PANJANG_CM))
    fs.write("gerbong_lebar_cm", float(GERBONG_LEBAR_CM))
    fs.write("gerbong_tinggi_cm", float(GERBONG_TINGGI_CM))

    fs.write("baseline_m", float(stereo_params["baseline_m"]))
    fs.write("baseline_mm", float(stereo_params["baseline_mm"]))
    if stereo_params["rms_stereo"] is not None:
        fs.write("rms_stereo", float(stereo_params["rms_stereo"]))
    if stereo_params["valid_pairs"] is not None:
        fs.write("valid_pairs", int(stereo_params["valid_pairs"]))

    fs.release()
    print(f"[OK] stereomap.xml tersimpan: {path_xml}")


def buat_atau_perbarui_stereomap():
    """
    Membuat stereomap.xml dari hasil kalibrasi ekstrinsik.
    Jika stereomap.xml sudah lengkap, bagian ini tetap dapat dipakai untuk memperbarui file tersebut.
    """
    print("\n[1] Memuat intrinsik dan parameter ekstrinsik...")
    intrinsik_kiri = muat_intrinsik(INTRINSIK_KIRI_XML, "kiri")
    intrinsik_kanan = muat_intrinsik(INTRINSIK_KANAN_XML, "kanan")
    stereo_params = muat_stereo_params(STEREO_PARAMS_XML)

    if intrinsik_kiri["image_size"] != intrinsik_kanan["image_size"]:
        raise RuntimeError(
            f"Ukuran intrinsik kiri {intrinsik_kiri['image_size']} tidak sama dengan kanan {intrinsik_kanan['image_size']}"
        )

    image_size = intrinsik_kiri["image_size"]

    cameraMatrixL = intrinsik_kiri["camera_matrix"]
    distL = intrinsik_kiri["dist_coeffs"]
    cameraMatrixR = intrinsik_kanan["camera_matrix"]
    distR = intrinsik_kanan["dist_coeffs"]

    R = stereo_params["R"]
    T = stereo_params["T"]
    E = stereo_params["E"]
    F = stereo_params["F"]

    print("\n[2] Rektifikasi stereo dengan cv2.stereoRectify...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrixL,
        distL,
        cameraMatrixR,
        distR,
        image_size,
        R,
        T,
        alpha=1
    )

    print(f"ROI valid kiri  : {roi1}")
    print(f"ROI valid kanan : {roi2}")
    print(f"Matriks Q:\n{Q}")

    print("\n[3] Inisialisasi stereo maps x/y dengan cv2.initUndistortRectifyMap...")
    mapLx, mapLy = cv2.initUndistortRectifyMap(
        cameraMatrixL,
        distL,
        R1,
        P1,
        image_size,
        cv2.CV_32FC1
    )

    mapRx, mapRy = cv2.initUndistortRectifyMap(
        cameraMatrixR,
        distR,
        R2,
        P2,
        image_size,
        cv2.CV_32FC1
    )

    print(f"Map kiri x/y  : {mapLx.shape} / {mapLy.shape}")
    print(f"Map kanan x/y : {mapRx.shape} / {mapRy.shape}")

    matrix_dict = buat_matrix_dict_dari_data(
        intrinsik_kiri=intrinsik_kiri,
        intrinsik_kanan=intrinsik_kanan,
        stereo_params=stereo_params,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        roi1=roi1,
        roi2=roi2,
        mapLx=mapLx,
        mapLy=mapLy,
        mapRx=mapRx,
        mapRy=mapRy
    )
    tampilkan_dan_simpan_semua_matriks(matrix_dict)

    print("\n[4] Menyimpan stereo maps dan parameter ke XML...")
    simpan_stereomap_xml(
        STEREOMAP_XML,
        mapLx,
        mapLy,
        mapRx,
        mapRy,
        Q,
        R,
        T,
        E,
        F,
        R1,
        R2,
        P1,
        P2,
        roi1,
        roi2,
        intrinsik_kiri,
        intrinsik_kanan,
        stereo_params,
        image_size
    )

    return muat_stereomap(STEREOMAP_XML)


# =========================================================
# REAL-TIME REMAP
# =========================================================
def resize_agar_muatt_layar(img, max_width=DISPLAY_MAX_WIDTH, max_height=DISPLAY_MAX_HEIGHT):
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img, 1.0

    scale = min(max_width / float(w), max_height / float(h), 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    if scale < 1.0:
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img, scale


def gambar_garis_epipolar(img, jumlah=15, warna=(0, 255, 0)):
    out = img.copy()
    h, w = out.shape[:2]
    for i in range(1, jumlah + 1):
        y = int(h * i / (jumlah + 1))
        cv2.line(out, (0, y), (w, y), warna, 1)
    return out


def tambah_label(img, teks):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, teks, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    return out


def mirror_untuk_tampilan(img):
    return cv2.flip(img, 1) if DISPLAY_MIRROR_HORIZONTAL else img


def pastikan_ukuran_frame(frame, image_size, nama_kamera, warning_state):
    target_w, target_h = image_size
    h, w = frame.shape[:2]

    if (w, h) == image_size:
        return frame, warning_state

    if not warning_state.get(nama_kamera, False):
        print(
            f"[WARNING] Ukuran frame {nama_kamera} = {(w, h)}, berbeda dari ukuran kalibrasi {image_size}."
        )
        if ALLOW_RESIZE_IF_NEEDED:
            print("          Frame akan di-resize untuk uji tampilan. Untuk hasil final, gunakan resolusi kamera yang sama dengan kalibrasi.")
        else:
            print("          Proses dihentikan. Atur resolusi kamera agar sama dengan kalibrasi.")
        warning_state[nama_kamera] = True

    if not ALLOW_RESIZE_IF_NEEDED:
        raise RuntimeError(f"Ukuran frame {nama_kamera} tidak sesuai kalibrasi.")

    return cv2.resize(frame, (target_w, target_h)), warning_state


def set_property_aman(cap, prop, value):
    try:
        cap.set(prop, value)
    except Exception:
        pass


def samakan_setting_kamera(cap, nama):
    if not SAMAKAN_SETTING_KAMERA:
        return

    set_property_aman(cap, cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS_VALUE)
    set_property_aman(cap, cv2.CAP_PROP_CONTRAST, CONTRAST_VALUE)
    set_property_aman(cap, cv2.CAP_PROP_GAIN, GAIN_VALUE)

    if hasattr(cv2, "CAP_PROP_AUTO_WB"):
        set_property_aman(cap, cv2.CAP_PROP_AUTO_WB, 0)

    if hasattr(cv2, "CAP_PROP_WB_TEMPERATURE"):
        set_property_aman(cap, cv2.CAP_PROP_WB_TEMPERATURE, WB_TEMPERATURE)

    if GUNAKAN_MANUAL_EXPOSURE and hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
        set_property_aman(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        set_property_aman(cap, cv2.CAP_PROP_EXPOSURE, EXPOSURE_VALUE)

    if KUNCI_AUTOFOCUS and hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
        set_property_aman(cap, cv2.CAP_PROP_AUTOFOCUS, 0)
        if str(nama).upper() == "KIRI":
            set_property_aman(cap, cv2.CAP_PROP_FOCUS, FOCUS_LEFT)
        else:
            set_property_aman(cap, cv2.CAP_PROP_FOCUS, FOCUS_RIGHT)


def buka_kamera(camera_id, image_size):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera ID {camera_id} tidak bisa dibuka.")

    # Kamera dibuka pada ukuran raw, lalu crop-resize dilakukan di software
    # agar sama dengan format gambar kalibrasi Script 06.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(CAMERA_RAW_WIDTH))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(CAMERA_RAW_HEIGHT))
    cap.set(cv2.CAP_PROP_FPS, int(FPS))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    nama_kamera = "KIRI" if int(camera_id) == int(CAMERA_ID_KIRI) else "KANAN"
    samakan_setting_kamera(cap, nama_kamera)
    return cap


def jalankan_realtime_remap(stereomap):
    os.makedirs(REALTIME_OUTPUT_DIR, exist_ok=True)
    os.makedirs(BEFORE_AFTER_OUTPUT_DIR, exist_ok=True)

    mapLx = stereomap["stereoMapL_x"]
    mapLy = stereomap["stereoMapL_y"]
    mapRx = stereomap["stereoMapR_x"]
    mapRy = stereomap["stereoMapR_y"]
    image_size = stereomap["image_size"]

    print("\n[5] Membuka kamera untuk real-time remap...")
    print(f"Kamera kiri  ID : {CAMERA_ID_KIRI}")
    print(f"Kamera kanan ID : {CAMERA_ID_KANAN}")
    print(f"Resolusi acuan  : {image_size}")

    capL = buka_kamera(CAMERA_ID_KIRI, image_size)
    capR = buka_kamera(CAMERA_ID_KANAN, image_size)

    print("\n[INFO] Kontrol:")
    print("       S   = simpan gambar sebelum, sesudah, dan output realtime")
    print("       G   = tampil/sembunyi garis epipolar")
    print("       ESC = keluar")
    print(f"       Tampilan otomatis diperkecil maksimal {DISPLAY_MAX_WIDTH}x{DISPLAY_MAX_HEIGHT} agar muat layar")

    show_lines = True
    screenshot_count = 0
    warning_state = {}
    auto_saved = False

    while True:
        retL, frameL_raw = capL.read()
        retR, frameR_raw = capR.read()

        if not retL or not retR:
            print("[ERROR] Frame kamera gagal dibaca.")
            break

        # Samakan format frame live dengan format gambar saat kalibrasi.
        # Jika kalibrasi memakai crop Script 06, crop ini wajib dilakukan sebelum remap.
        frameL = persiapkan_frame_realtime(frameL_raw, "kiri", image_size)
        frameR = persiapkan_frame_realtime(frameR_raw, "kanan", image_size)

        frameL, warning_state = pastikan_ukuran_frame(frameL, image_size, "kiri", warning_state)
        frameR, warning_state = pastikan_ukuran_frame(frameR, image_size, "kanan", warning_state)

        rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

        if show_lines:
            rectL_simpan = gambar_garis_epipolar(rectL)
            rectR_simpan = gambar_garis_epipolar(rectR)
        else:
            rectL_simpan = rectL.copy()
            rectR_simpan = rectR.copy()

        sebelumL = tambah_label(mirror_untuk_tampilan(frameL.copy()), "KIRI - SEBELUM REKTIFIKASI")
        sebelumR = tambah_label(mirror_untuk_tampilan(frameR.copy()), "KANAN - SEBELUM REKTIFIKASI")
        rectL_tampil = tambah_label(mirror_untuk_tampilan(rectL_simpan), "KIRI - SETELAH REKTIFIKASI / cv2.remap")
        rectR_tampil = tambah_label(mirror_untuk_tampilan(rectR_simpan), "KANAN - SETELAH REKTIFIKASI / cv2.remap")

        baris_sebelum = np.hstack([sebelumL, sebelumR])
        baris_sesudah = np.hstack([rectL_tampil, rectR_tampil])

        label_sebelum = np.zeros((35, baris_sebelum.shape[1], 3), dtype=np.uint8)
        label_sesudah = np.zeros((35, baris_sesudah.shape[1], 3), dtype=np.uint8)

        cv2.putText(label_sebelum, "GAMBAR SEBELUM REKTIFIKASI", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
        cv2.putText(label_sesudah, "GAMBAR SETELAH REKTIFIKASI - OUTPUT REALTIME", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        gabung = np.vstack([label_sebelum, baris_sebelum, label_sesudah, baris_sesudah])

        baris_sebelum_simpan = np.hstack([
            tambah_label(frameL.copy(), "KIRI - SEBELUM REKTIFIKASI"),
            tambah_label(frameR.copy(), "KANAN - SEBELUM REKTIFIKASI")
        ])
        baris_sesudah_simpan = np.hstack([
            tambah_label(rectL_simpan.copy(), "KIRI - SETELAH REKTIFIKASI / cv2.remap"),
            tambah_label(rectR_simpan.copy(), "KANAN - SETELAH REKTIFIKASI / cv2.remap")
        ])
        gabung_simpan = np.vstack([label_sebelum, baris_sebelum_simpan, label_sesudah, baris_sesudah_simpan])

        gabung_display, scale = resize_agar_muatt_layar(gabung)

        if DISPLAY_WINDOW_NORMAL:
            cv2.namedWindow(WINDOW_NAME_REALTIME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME_REALTIME, gabung_display.shape[1], gabung_display.shape[0])

        cv2.imshow(WINDOW_NAME_REALTIME, gabung_display)

        if AUTO_SAVE_FIRST_FRAME and not auto_saved:
            screenshot_count += 1
            simpan_gambar_sebelum_sesudah(frameL, frameR, rectL_simpan, rectR_simpan, gabung_simpan, screenshot_count)
            auto_saved = True

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == ord("g") or key == ord("G"):
            show_lines = not show_lines
        elif key == ord("s") or key == ord("S"):
            screenshot_count += 1
            simpan_gambar_sebelum_sesudah(frameL, frameR, rectL_simpan, rectR_simpan, gabung_simpan, screenshot_count)

            waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
            path_out = os.path.join(REALTIME_OUTPUT_DIR, f"remap_{screenshot_count:03d}_{waktu}.png")
            cv2.imwrite(path_out, gabung_simpan)
            print(f"[OK] Screenshot realtime tersimpan: {path_out}")

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

    print(f"\n[SELESAI] Screenshot real-time tersimpan di: {REALTIME_OUTPUT_DIR}")
    print(f"[SELESAI] Gambar sebelum/sesudah rektifikasi tersimpan di: {BEFORE_AFTER_OUTPUT_DIR}")



# =========================================================
# MAIN
# =========================================================
def main():
    print("\n" + "=" * 72)
    print("SCRIPT 4 - REKTIFIKASI + STEREOMAP + REAL-TIME REMAP")
    print("SELARAS DENGAN SCRIPT 06 + SCRIPT 2 + SCRIPT 3 CROP MULTI DETECT")
    print("=" * 72)
    print(f"Intrinsik kiri  : {INTRINSIK_KIRI_XML}")
    print(f"Intrinsik kanan : {INTRINSIK_KANAN_XML}")
    print(f"Stereo params   : {STEREO_PARAMS_XML}")
    print(f"Stereomap XML   : {STEREOMAP_XML}")
    print("=" * 72)

    # Jika stereomap.xml sudah memiliki maps lengkap, langsung muat.
    # Jika belum ada atau belum lengkap, buat ulang dari stereo_params.xml + intrinsik XML.
    stereomap_lengkap = (
        cek_node_ada(STEREOMAP_XML, "stereoMapL_x") and
        cek_node_ada(STEREOMAP_XML, "stereoMapL_y") and
        cek_node_ada(STEREOMAP_XML, "stereoMapR_x") and
        cek_node_ada(STEREOMAP_XML, "stereoMapR_y")
    )

    if FORCE_REGENERATE_STEREOMAP:
        print("\n[INFO] FORCE_REGENERATE_STEREOMAP=True. Membuat ulang stereomap agar semua matriks rektifikasi tampil.")
        stereomap = buat_atau_perbarui_stereomap()
    elif stereomap_lengkap:
        print("\n[INFO] stereomap.xml sudah lengkap. File akan langsung dimuat.")
        stereomap = muat_stereomap(STEREOMAP_XML)

        matrix_dict = buat_matrix_dict_dari_data(stereomap=stereomap)
        tampilkan_dan_simpan_semua_matriks(matrix_dict)
    else:
        print("\n[INFO] stereomap.xml belum ada atau belum lengkap. Membuat ulang stereomap.xml...")
        stereomap = buat_atau_perbarui_stereomap()

    print("\n[INFO] Sistem siap menggunakan cv2.remap secara real-time.")
    jalankan_realtime_remap(stereomap)


if __name__ == "__main__":
    main()
