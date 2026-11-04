"""
SCRIPT 3 - KALIBRASI EKSTRINSIK/STEREO CHARUCO + REKTIFIKASI + STEREOMAP XML
================================================================================

Fungsi:
1. Memuat hasil kalibrasi intrinsik dari Script 2 dalam format XML:
   - hasil_kalibrasi/intrinsik_kiri.xml
   - hasil_kalibrasi/intrinsik_kanan.xml
2. Membaca pasangan gambar ChArUco dari:
   - charuco_left
   - charuco_right
3. Mendeteksi titik ChArUco yang sama pada kamera kiri dan kanan.
4. Melakukan kalibrasi ekstrinsik/stereo dengan cv2.stereoCalibrate().
5. Menggunakan cv2.CALIB_FIX_INTRINSIC agar nilai intrinsik dikunci.
6. Menghasilkan parameter stereo: R, T, E, F.
7. Melakukan stereo rectification dengan cv2.stereoRectify().
8. Membuat stereo map dengan cv2.initUndistortRectifyMap().
9. Menyimpan hasil akhir ke:
   - hasil_kalibrasi/stereomap.xml
10. Menyimpan preview rektifikasi dan ringkasan gambar pendukung.

Catatan penting:
- Script ini sudah diselaraskan dengan Script 2 ChArUco XML.
- Script ini TIDAK membaca file .npz.
- Script ini TIDAK memakai chessboard biasa.
- Script ini memakai ChArUco sesuai folder charuco_left dan charuco_right.

Cara pakai:
python 03_kalibrasi_stereo_CHARUCO_XML_SELARAS_06_CROP_MULTI_DETECT.py
"""

import cv2
from cv2 import aruco
import numpy as np
import glob
import os

# Matplotlib hanya dipakai untuk menyimpan grafik ringkasan.
# Jika tidak tersedia, proses kalibrasi tetap berjalan dan hanya grafik ringkasan yang dilewati.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    plt = None
    MATPLOTLIB_OK = False


# =========================================================
# KONFIGURASI CHARUCO BOARD - HARUS SAMA DENGAN SCRIPT 2
# =========================================================
SQUARES_X = 5
SQUARES_Y = 7

SQUARE_LENGTH = 0.039   # meter, sama dengan 39 mm
MARKER_LENGTH = 0.029   # meter, sama dengan 29 mm

MIN_COMMON_CHARUCO_CORNERS = 6
ARUCO_DICT_ID = aruco.DICT_4X4_50
ARUCO_DICT_NAME = "DICT_4X4_50"


# =========================================================
# KONFIGURASI SELARAS SCRIPT 06 CAPTURE + SCRIPT 2 INTRINSIK
# =========================================================
WIDTH = 640
HEIGHT = 480

# Script 06 kamu menyimpan frame crop ke charuco_left/charuco_right.
# Karena itu default-nya input dianggap sudah crop, sehingga tidak di-crop ulang.
# Jika suatu saat folder berisi gambar full frame, ubah INPUT_SUDAH_CROP_DARI_SCRIPT_06=False
# dan TERAPKAN_CROP_ULANG=True.
CROP_AKTIF = True
SIMPAN_FRAME_CROP = True
INPUT_SUDAH_CROP_DARI_SCRIPT_06 = True
TERAPKAN_CROP_ULANG = False

ROI_LEFT = (0.00, 0.05, 1.00, 0.95)
ROI_RIGHT = (0.00, 0.05, 1.00, 0.95)

CAMERA_HEIGHT_CM = 60.0
CAMERA_TILT_MIN_DEG = 10.0
CAMERA_TILT_MAX_DEG = 15.0
CAMERA_FOV_DEG = 120.0

GERBONG_PANJANG_CM = 40.0
GERBONG_LEBAR_CM = 10.0
GERBONG_TINGGI_CM = 7.5

GUNAKAN_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

GUNAKAN_SHARPEN_RINGAN = True
SHARPEN_SIGMA = 1.0
SHARPEN_AMOUNT = 0.55

GUNAKAN_MULTI_DETECT = True
LEFT_EXTRA_BOOST = True
LEFT_CONTRAST_ALPHA = 1.35
LEFT_BRIGHTNESS_BETA = 8
LEFT_SHARPEN_AMOUNT_EXTRA = 0.85

MIN_SHARPNESS_LEFT = 35.0
MIN_SHARPNESS_RIGHT = 35.0


# =========================================================
# KONFIGURASI FOLDER
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FOLDER_LEFT = os.path.join(BASE_DIR, "charuco_left")
FOLDER_RIGHT = os.path.join(BASE_DIR, "charuco_right")

OUTPUT_DIR = os.path.join(BASE_DIR, "hasil_kalibrasi")
INPUT_INTRINSIK_KIRI = os.path.join(OUTPUT_DIR, "intrinsik_kiri.xml")
INPUT_INTRINSIK_KANAN = os.path.join(OUTPUT_DIR, "intrinsik_kanan.xml")

OUTPUT_STEREOMAP_XML = os.path.join(OUTPUT_DIR, "stereomap.xml")
OUTPUT_STEREO_PARAMS_XML = os.path.join(OUTPUT_DIR, "stereo_params.xml")

PREVIEW_RECTIFIED_DIR = os.path.join(OUTPUT_DIR, "preview_stereo_rectified")
VISUAL_STEREO_DIR = os.path.join(OUTPUT_DIR, "visualisasi_stereo")


# =========================================================
# VALIDASI MODUL ARUCO
# =========================================================
def cek_aruco():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "Modul cv2.aruco tidak ditemukan.\n"
            "Install OpenCV contrib dengan perintah:\n"
            "pip install opencv-contrib-python"
        )

    if not hasattr(aruco, "CharucoBoard") and not hasattr(aruco, "CharucoBoard_create"):
        raise RuntimeError(
            "OpenCV yang terpasang belum mendukung ChArUco dengan lengkap.\n"
            "Install ulang dengan:\n"
            "pip install --upgrade opencv-contrib-python"
        )


# =========================================================
# UTILITAS FILESTORAGE XML
# =========================================================
def _node_empty(node):
    try:
        return node.empty()
    except Exception:
        return True


def baca_matrix(fs, nama_node, wajib=True):
    node = fs.getNode(nama_node)
    if _node_empty(node):
        if wajib:
            raise RuntimeError(f"Node '{nama_node}' tidak ditemukan di file XML.")
        return None
    mat = node.mat()
    if mat is None and wajib:
        raise RuntimeError(f"Node '{nama_node}' gagal dibaca sebagai matrix.")
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


def muat_intrinsik_xml(nama_kamera, path_xml):
    if not os.path.exists(path_xml):
        raise FileNotFoundError(
            f"File intrinsik kamera {nama_kamera} tidak ditemukan:\n{path_xml}\n"
            "Jalankan dulu Script 2 kalibrasi intrinsik ChArUco XML."
        )

    fs = cv2.FileStorage(path_xml, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Gagal membuka file XML: {path_xml}")

    camera_matrix = baca_matrix(fs, "camera_matrix")
    dist_coeffs = baca_matrix(fs, "dist_coeffs")

    # Script 2 versi terbaru menyimpan image_width dan image_height.
    # Beberapa versi awal mungkin menyimpan image_size sebagai matrix.
    image_width = baca_int(fs, "image_width", default=None)
    image_height = baca_int(fs, "image_height", default=None)

    if image_width is None or image_height is None:
        image_size_mat = baca_matrix(fs, "image_size", wajib=False)
        if image_size_mat is None:
            raise RuntimeError(
                f"Ukuran gambar tidak ditemukan di {path_xml}. "
                "Pastikan XML hasil intrinsik menyimpan image_width/image_height atau image_size."
            )
        image_size_flat = np.asarray(image_size_mat).astype(int).ravel()
        image_width = int(image_size_flat[0])
        image_height = int(image_size_flat[1])

    mean_error = baca_float(fs, "mean_error", default=-1.0)
    rms = baca_float(fs, "rms", default=-1.0)
    pola_terpilih = baca_string(fs, "pola_terpilih", default="DEFAULT")

    # Metadata tambahan dari Script 2 yang sudah diselaraskan dengan Script 06.
    roi_capture = baca_matrix(fs, "roi_capture", wajib=False)
    input_sudah_crop = baca_int(fs, "input_sudah_crop_dari_script_06", default=None)
    terapkan_crop_ulang_xml = baca_int(fs, "terapkan_crop_ulang", default=None)
    gunakan_multi_detect_xml = baca_int(fs, "gunakan_multi_detect", default=None)

    fs.release()

    print(f"[OK] Intrinsik {nama_kamera} dimuat: {path_xml}")
    print(f"     image size : {(image_width, image_height)}")
    print(f"     mean error : {mean_error:.6f} px" if mean_error >= 0 else "     mean error : tidak tersedia")
    print(f"     RMS        : {rms:.6f} px" if rms >= 0 else "     RMS        : tidak tersedia")
    print(f"     pola       : {pola_terpilih}")
    if roi_capture is not None:
        print(f"     roi capture: {np.asarray(roi_capture).ravel().tolist()}")
    if input_sudah_crop is not None:
        print(f"     input crop : {bool(input_sudah_crop)} | crop ulang XML: {bool(terapkan_crop_ulang_xml) if terapkan_crop_ulang_xml is not None else 'tidak tersedia'}")

    return {
        "camera_matrix": camera_matrix.astype(np.float64),
        "dist_coeffs": dist_coeffs.astype(np.float64),
        "image_size": (image_width, image_height),
        "mean_error": mean_error,
        "rms": rms,
        "pola_terpilih": pola_terpilih,
        "path_xml": path_xml,
        "roi_capture": roi_capture,
        "input_sudah_crop_dari_script_06": input_sudah_crop,
        "terapkan_crop_ulang_xml": terapkan_crop_ulang_xml,
        "gunakan_multi_detect_xml": gunakan_multi_detect_xml,
    }


# =========================================================
# PREPROCESSING / ROI - SELARAS SCRIPT 06
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


def crop_resize_frame(frame, roi, output_size=(WIDTH, HEIGHT)):
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


def persiapkan_gambar_untuk_stereo(img, nama_kamera):
    if img is None:
        return None

    # Jika gambar berasal dari Script 06 dengan SIMPAN_FRAME_CROP=True,
    # gambar di folder charuco_left/right sudah crop-resize. Jangan crop ulang.
    if TERAPKAN_CROP_ULANG and not INPUT_SUDAH_CROP_DARI_SCRIPT_06:
        return crop_resize_frame(img, pilih_roi(nama_kamera))

    return img


def hitung_sharpness(gray):
    if gray is None:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sharpen_gray(gray, amount=0.55, sigma=1.0):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)


def preprocess_gray(gray):
    if gray is None:
        return gray

    if GUNAKAN_CLAHE:
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID
        )
        gray = clahe.apply(gray)

    if GUNAKAN_SHARPEN_RINGAN:
        gray = sharpen_gray(gray, amount=SHARPEN_AMOUNT, sigma=SHARPEN_SIGMA)

    return gray


def boost_left_gray(gray):
    boosted = cv2.convertScaleAbs(
        gray,
        alpha=LEFT_CONTRAST_ALPHA,
        beta=LEFT_BRIGHTNESS_BETA
    )
    boosted = sharpen_gray(
        boosted,
        amount=LEFT_SHARPEN_AMOUNT_EXTRA,
        sigma=SHARPEN_SIGMA
    )
    return boosted


def buat_variasi_gray(gray, nama_kamera):
    variasi = [("RAW", gray)]

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )
    gray_clahe = clahe.apply(gray)
    variasi.append(("CLAHE", gray_clahe))

    variasi.append(("CLAHE_SHARP", sharpen_gray(
        gray_clahe,
        amount=SHARPEN_AMOUNT,
        sigma=SHARPEN_SIGMA
    )))

    variasi.append(("EQUALIZE", cv2.equalizeHist(gray)))
    variasi.append(("CONTRAST", cv2.convertScaleAbs(gray, alpha=1.25, beta=5)))

    if LEFT_EXTRA_BOOST and str(nama_kamera).lower() in ["kiri", "left"]:
        variasi.append(("LEFT_BOOST", boost_left_gray(gray)))

    return variasi


def skor_hasil_deteksi(hasil):
    return (hasil["jumlah_charuco"] * 1000) + hasil["jumlah_marker"]


# =========================================================
# MEMBUAT CHARUCO BOARD
# =========================================================
def buat_charuco_board(legacy=False):
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_ID)

    if hasattr(aruco, "CharucoBoard"):
        try:
            board = aruco.CharucoBoard(
                (SQUARES_X, SQUARES_Y),
                SQUARE_LENGTH,
                MARKER_LENGTH,
                aruco_dict
            )
        except TypeError:
            board = aruco.CharucoBoard_create(
                SQUARES_X,
                SQUARES_Y,
                SQUARE_LENGTH,
                MARKER_LENGTH,
                aruco_dict
            )
    else:
        board = aruco.CharucoBoard_create(
            SQUARES_X,
            SQUARES_Y,
            SQUARE_LENGTH,
            MARKER_LENGTH,
            aruco_dict
        )

    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(legacy)

    return board, aruco_dict


def buat_charuco_detector(board):
    if hasattr(aruco, "CharucoDetector"):
        return aruco.CharucoDetector(board)
    return None


def parse_hasil_detect_board(hasil):
    charuco_corners = None
    charuco_ids = None
    marker_corners = None
    marker_ids = None

    if hasil is None or not isinstance(hasil, tuple):
        return charuco_corners, charuco_ids, marker_corners, marker_ids

    if len(hasil) == 4:
        charuco_corners, charuco_ids, marker_corners, marker_ids = hasil
    elif len(hasil) == 3:
        a, b, c = hasil
        if isinstance(a, (bool, int, float, np.bool_)):
            charuco_corners = b
            charuco_ids = c
        else:
            charuco_corners = a
            charuco_ids = b
            marker_corners = c
    elif len(hasil) == 2:
        charuco_corners, charuco_ids = hasil

    return charuco_corners, charuco_ids, marker_corners, marker_ids


def buat_detector_parameters():
    if hasattr(aruco, "DetectorParameters"):
        return aruco.DetectorParameters()
    return aruco.DetectorParameters_create()


def deteksi_charuco_legacy_api(gray, board, aruco_dict):
    parameters = buat_detector_parameters()

    marker_corners, marker_ids, _ = aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    if marker_ids is None or len(marker_ids) == 0:
        return None, None, None, None

    _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board
    )

    return charuco_corners, charuco_ids, marker_corners, marker_ids


def detect_charuco(gray, board, aruco_dict):
    detector = buat_charuco_detector(board)

    if detector is not None:
        hasil = detector.detectBoard(gray)
        charuco_corners, charuco_ids, marker_corners, marker_ids = parse_hasil_detect_board(hasil)
    else:
        charuco_corners, charuco_ids, marker_corners, marker_ids = deteksi_charuco_legacy_api(
            gray,
            board,
            aruco_dict
        )

    jumlah_charuco = 0 if charuco_ids is None else len(charuco_ids)
    jumlah_marker = 0 if marker_ids is None else len(marker_ids)

    return {
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "jumlah_charuco": jumlah_charuco,
        "jumlah_marker": jumlah_marker
    }


def detect_charuco_multi(gray, nama_kamera, board, aruco_dict):
    if not GUNAKAN_MULTI_DETECT:
        hasil = detect_charuco(preprocess_gray(gray), board, aruco_dict)
        hasil["variasi"] = "NORMAL"
        return hasil

    hasil_terbaik = None
    nama_terbaik = None
    skor_terbaik = -1

    for nama_variasi, gray_variasi in buat_variasi_gray(gray, nama_kamera):
        hasil = detect_charuco(gray_variasi, board, aruco_dict)
        skor = skor_hasil_deteksi(hasil)
        if skor > skor_terbaik:
            hasil_terbaik = hasil
            nama_terbaik = nama_variasi
            skor_terbaik = skor

    hasil_terbaik["variasi"] = nama_terbaik
    return hasil_terbaik


# =========================================================
# OBJECT POINT CHARUCO
# =========================================================
def ambil_semua_objek_corner_charuco(board):
    if hasattr(board, "getChessboardCorners"):
        corners_3d = board.getChessboardCorners()
    else:
        corners_3d = board.chessboardCorners

    return np.asarray(corners_3d, dtype=np.float32)


def ambil_common_points_dari_deteksi(board, hasil_kiri, hasil_kanan):
    """
    Membuat objpoints, imgpoints kiri, dan imgpoints kanan dari ID ChArUco yang sama.
    Urutan titik harus sama di kiri dan kanan agar stereoCalibrate benar.
    """
    corners_kiri = hasil_kiri["charuco_corners"]
    ids_kiri = hasil_kiri["charuco_ids"]
    corners_kanan = hasil_kanan["charuco_corners"]
    ids_kanan = hasil_kanan["charuco_ids"]

    if corners_kiri is None or ids_kiri is None or corners_kanan is None or ids_kanan is None:
        return None, None, None, []

    ids_kiri_flat = ids_kiri.flatten().astype(int)
    ids_kanan_flat = ids_kanan.flatten().astype(int)

    common_ids = sorted(list(set(ids_kiri_flat.tolist()) & set(ids_kanan_flat.tolist())))

    if len(common_ids) < MIN_COMMON_CHARUCO_CORNERS:
        return None, None, None, common_ids

    semua_objek_corner = ambil_semua_objek_corner_charuco(board)

    objpoints = []
    imgpoints_kiri = []
    imgpoints_kanan = []

    for cid in common_ids:
        idx_kiri = int(np.where(ids_kiri_flat == cid)[0][0])
        idx_kanan = int(np.where(ids_kanan_flat == cid)[0][0])

        objpoints.append(semua_objek_corner[cid])
        imgpoints_kiri.append(corners_kiri[idx_kiri].reshape(2))
        imgpoints_kanan.append(corners_kanan[idx_kanan].reshape(2))

    objpoints = np.asarray(objpoints, dtype=np.float32).reshape(-1, 1, 3)
    imgpoints_kiri = np.asarray(imgpoints_kiri, dtype=np.float32).reshape(-1, 1, 2)
    imgpoints_kanan = np.asarray(imgpoints_kanan, dtype=np.float32).reshape(-1, 1, 2)

    return objpoints, imgpoints_kiri, imgpoints_kanan, common_ids


# =========================================================
# FILE GAMBAR PASANGAN
# =========================================================
def ambil_daftar_gambar(folder):
    pola = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    daftar = []
    for p in pola:
        daftar.extend(glob.glob(os.path.join(folder, p)))
    return sorted(daftar)


def ambil_pasangan_gambar():
    gambar_kiri = ambil_daftar_gambar(FOLDER_LEFT)
    gambar_kanan = ambil_daftar_gambar(FOLDER_RIGHT)

    dict_kiri = {os.path.basename(p): p for p in gambar_kiri}
    dict_kanan = {os.path.basename(p): p for p in gambar_kanan}

    nama_sama = sorted(list(set(dict_kiri.keys()) & set(dict_kanan.keys())))

    pasangan = [(nama, dict_kiri[nama], dict_kanan[nama]) for nama in nama_sama]

    print(f"[INFO] Jumlah gambar kiri  : {len(gambar_kiri)}")
    print(f"[INFO] Jumlah gambar kanan : {len(gambar_kanan)}")
    print(f"[INFO] Jumlah nama pasangan sama: {len(pasangan)}")

    if len(pasangan) == 0:
        raise RuntimeError(
            "Tidak ada pasangan gambar dengan nama file yang sama.\n"
            "Pastikan folder charuco_left dan charuco_right berisi frame dengan nama sama, "
            "misalnya frame_001.png pada kedua folder."
        )

    return pasangan


# =========================================================
# KUMPULKAN STEREO POINTS CHARUCO
# =========================================================
def kumpulkan_stereo_points_charuco(board, aruco_dict, image_size_intrinsik):
    pasangan = ambil_pasangan_gambar()

    objpoints_all = []
    imgpoints_kiri_all = []
    imgpoints_kanan_all = []
    valid_pairs = []
    image_size_dari_gambar = None

    print("\n[2] Mendeteksi ChArUco pada pasangan gambar stereo...")

    for idx, (nama_file, path_kiri, path_kanan) in enumerate(pasangan, 1):
        img_kiri = cv2.imread(path_kiri)
        img_kanan = cv2.imread(path_kanan)

        if img_kiri is None or img_kanan is None:
            print(f"[{idx:03d}] DILEWATI: {nama_file} - gagal membaca gambar")
            continue

        img_kiri = persiapkan_gambar_untuk_stereo(img_kiri, "kiri")
        img_kanan = persiapkan_gambar_untuk_stereo(img_kanan, "kanan")

        gray_kiri = cv2.cvtColor(img_kiri, cv2.COLOR_BGR2GRAY)
        gray_kanan = cv2.cvtColor(img_kanan, cv2.COLOR_BGR2GRAY)

        sharp_kiri = hitung_sharpness(gray_kiri)
        sharp_kanan = hitung_sharpness(gray_kanan)

        ukuran_kiri = (gray_kiri.shape[1], gray_kiri.shape[0])
        ukuran_kanan = (gray_kanan.shape[1], gray_kanan.shape[0])

        if ukuran_kiri != ukuran_kanan:
            print(f"[{idx:03d}] DILEWATI: {nama_file} - ukuran kiri {ukuran_kiri}, kanan {ukuran_kanan}")
            continue

        if ukuran_kiri != image_size_intrinsik:
            print(
                f"[{idx:03d}] DILEWATI: {nama_file} - ukuran gambar {ukuran_kiri}, "
                f"harus sama dengan intrinsik {image_size_intrinsik}"
            )
            continue

        if image_size_dari_gambar is None:
            image_size_dari_gambar = ukuran_kiri

        hasil_kiri = detect_charuco_multi(gray_kiri, "kiri", board, aruco_dict)
        hasil_kanan = detect_charuco_multi(gray_kanan, "kanan", board, aruco_dict)

        objp, imgp_kiri, imgp_kanan, common_ids = ambil_common_points_dari_deteksi(
            board,
            hasil_kiri,
            hasil_kanan
        )

        if objp is None:
            print(
                f"[{idx:03d}] TIDAK VALID: {nama_file} | "
                f"kiri corners {hasil_kiri['jumlah_charuco']} | "
                f"kanan corners {hasil_kanan['jumlah_charuco']} | "
                f"common {len(common_ids)} | deteksi L/R {hasil_kiri.get('variasi', '-')} / {hasil_kanan.get('variasi', '-')} | sharp L/R {sharp_kiri:.1f} / {sharp_kanan:.1f}"
            )
            continue

        objpoints_all.append(objp)
        imgpoints_kiri_all.append(imgp_kiri)
        imgpoints_kanan_all.append(imgp_kanan)
        valid_pairs.append({
            "nama_file": nama_file,
            "path_kiri": path_kiri,
            "path_kanan": path_kanan,
            "jumlah_common": len(common_ids),
            "common_ids": common_ids,
            "deteksi_kiri": hasil_kiri.get("variasi", "-"),
            "deteksi_kanan": hasil_kanan.get("variasi", "-"),
            "sharp_kiri": sharp_kiri,
            "sharp_kanan": sharp_kanan,
        })

        print(
            f"[{idx:03d}] VALID     : {nama_file} | "
            f"common corners: {len(common_ids)} | "
            f"kiri: {hasil_kiri['jumlah_charuco']} ({hasil_kiri.get('variasi', '-')}, sharp {sharp_kiri:.1f}) | "
            f"kanan: {hasil_kanan['jumlah_charuco']} ({hasil_kanan.get('variasi', '-')}, sharp {sharp_kanan:.1f})"
        )

    print("\n[INFO] Ringkasan stereo point pairs")
    print(f"       Pasangan valid: {len(valid_pairs)} dari {len(pasangan)}")

    return objpoints_all, imgpoints_kiri_all, imgpoints_kanan_all, image_size_dari_gambar, valid_pairs


# =========================================================
# SIMPAN HASIL STEREO KE XML
# =========================================================
def simpan_stereomap_xml(
    path_xml,
    stereoMapL_x,
    stereoMapL_y,
    stereoMapR_x,
    stereoMapR_y,
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
    baseline_m,
    baseline_mm,
    rms_stereo,
    image_size,
    valid_count,
    intrinsic_kiri,
    intrinsic_kanan
):
    os.makedirs(os.path.dirname(path_xml), exist_ok=True)

    fs = cv2.FileStorage(path_xml, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Gagal membuat file XML: {path_xml}")

    # Stereo maps untuk cv2.remap pada aplikasi real-time.
    fs.write("stereoMapL_x", stereoMapL_x)
    fs.write("stereoMapL_y", stereoMapL_y)
    fs.write("stereoMapR_x", stereoMapR_x)
    fs.write("stereoMapR_y", stereoMapR_y)

    # Parameter ekstrinsik dan rectification.
    fs.write("R", R)
    fs.write("T", T)
    fs.write("E", E)
    fs.write("F", F)
    fs.write("R1", R1)
    fs.write("R2", R2)
    fs.write("P1", P1)
    fs.write("P2", P2)
    fs.write("Q", Q)

    # Parameter intrinsik ikut disimpan agar file stereomap.xml lengkap dan mudah dipakai ulang.
    fs.write("cameraMatrixL", intrinsic_kiri["camera_matrix"])
    fs.write("distCoeffsL", intrinsic_kiri["dist_coeffs"])
    fs.write("cameraMatrixR", intrinsic_kanan["camera_matrix"])
    fs.write("distCoeffsR", intrinsic_kanan["dist_coeffs"])

    # Metadata.
    fs.write("roi1", np.array(roi1, dtype=np.int32))
    fs.write("roi2", np.array(roi2, dtype=np.int32))
    fs.write("baseline_m", float(baseline_m))
    fs.write("baseline_mm", float(baseline_mm))
    fs.write("rms_stereo", float(rms_stereo))
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("valid_pairs", int(valid_count))
    fs.write("intrinsic_mean_error_left", float(intrinsic_kiri["mean_error"]))
    fs.write("intrinsic_mean_error_right", float(intrinsic_kanan["mean_error"]))
    fs.write("square_length_m", float(SQUARE_LENGTH))
    fs.write("marker_length_m", float(MARKER_LENGTH))
    fs.write("aruco_dict_name", str(ARUCO_DICT_NAME))
    fs.write("capture_script_reference", "06_CAPTURE_CHARUCO_STEREO_LEFT_BOOST_MULTI_DETECT")
    fs.write("script2_reference", "02_kalibrasi_intrinsik_charuco_SESUAI_06_CROP_MULTI_DETECT")
    fs.write("crop_aktif", int(CROP_AKTIF))
    fs.write("simpan_frame_crop", int(SIMPAN_FRAME_CROP))
    fs.write("input_sudah_crop_dari_script_06", int(INPUT_SUDAH_CROP_DARI_SCRIPT_06))
    fs.write("terapkan_crop_ulang", int(TERAPKAN_CROP_ULANG))
    fs.write("roi_left_capture", np.array(ROI_LEFT, dtype=np.float32).reshape(1, 4))
    fs.write("roi_right_capture", np.array(ROI_RIGHT, dtype=np.float32).reshape(1, 4))
    fs.write("gunakan_clahe", int(GUNAKAN_CLAHE))
    fs.write("gunakan_sharpen_ringan", int(GUNAKAN_SHARPEN_RINGAN))
    fs.write("gunakan_multi_detect", int(GUNAKAN_MULTI_DETECT))
    fs.write("left_extra_boost", int(LEFT_EXTRA_BOOST))
    fs.write("camera_height_cm", float(CAMERA_HEIGHT_CM))
    fs.write("camera_tilt_min_deg", float(CAMERA_TILT_MIN_DEG))
    fs.write("camera_tilt_max_deg", float(CAMERA_TILT_MAX_DEG))
    fs.write("camera_fov_deg", float(CAMERA_FOV_DEG))
    fs.write("gerbong_panjang_cm", float(GERBONG_PANJANG_CM))
    fs.write("gerbong_lebar_cm", float(GERBONG_LEBAR_CM))
    fs.write("gerbong_tinggi_cm", float(GERBONG_TINGGI_CM))

    fs.release()
    print(f"[OK] stereomap XML tersimpan: {path_xml}")


def simpan_stereo_params_xml(
    path_xml,
    Q,
    R,
    T,
    E,
    F,
    R1,
    R2,
    P1,
    P2,
    baseline_m,
    baseline_mm,
    rms_stereo,
    image_size,
    valid_count
):
    fs = cv2.FileStorage(path_xml, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Gagal membuat file XML: {path_xml}")

    fs.write("Q", Q)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("E", E)
    fs.write("F", F)
    fs.write("R1", R1)
    fs.write("R2", R2)
    fs.write("P1", P1)
    fs.write("P2", P2)
    fs.write("baseline_m", float(baseline_m))
    fs.write("baseline_mm", float(baseline_mm))
    fs.write("rms_stereo", float(rms_stereo))
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("valid_pairs", int(valid_count))
    fs.write("crop_aktif", int(CROP_AKTIF))
    fs.write("simpan_frame_crop", int(SIMPAN_FRAME_CROP))
    fs.write("input_sudah_crop_dari_script_06", int(INPUT_SUDAH_CROP_DARI_SCRIPT_06))
    fs.write("terapkan_crop_ulang", int(TERAPKAN_CROP_ULANG))
    fs.write("roi_left_capture", np.array(ROI_LEFT, dtype=np.float32).reshape(1, 4))
    fs.write("roi_right_capture", np.array(ROI_RIGHT, dtype=np.float32).reshape(1, 4))
    fs.write("gunakan_multi_detect", int(GUNAKAN_MULTI_DETECT))
    fs.release()
    print(f"[OK] stereo params XML tersimpan: {path_xml}")


# =========================================================
# PREVIEW REKTIFIKASI
# =========================================================
def gambar_garis_horizontal(img, step=40):
    out = img.copy()
    h, w = out.shape[:2]
    for y in range(step, h, step):
        cv2.line(out, (0, y), (w, y), (0, 255, 0), 1)
    return out


def simpan_preview_rektifikasi(valid_pairs, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y, max_preview=8):
    os.makedirs(PREVIEW_RECTIFIED_DIR, exist_ok=True)

    if len(valid_pairs) == 0:
        print("[WARNING] Tidak ada pasangan valid untuk preview rektifikasi.")
        return []

    # Ambil beberapa contoh dari awal, tengah, dan akhir agar preview tidak hanya satu kondisi.
    if len(valid_pairs) <= max_preview:
        indeks = list(range(len(valid_pairs)))
    else:
        indeks = sorted(set(np.linspace(0, len(valid_pairs) - 1, max_preview).astype(int).tolist()))

    output_paths = []

    for n, idx in enumerate(indeks, 1):
        item = valid_pairs[idx]
        imgL = cv2.imread(item["path_kiri"])
        imgR = cv2.imread(item["path_kanan"])

        if imgL is None or imgR is None:
            continue

        imgL = persiapkan_gambar_untuk_stereo(imgL, "kiri")
        imgR = persiapkan_gambar_untuk_stereo(imgR, "kanan")

        rectL = cv2.remap(imgL, stereoMapL_x, stereoMapL_y, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, stereoMapR_x, stereoMapR_y, cv2.INTER_LINEAR)

        gabung = np.hstack([rectL, rectR])
        gabung = gambar_garis_horizontal(gabung, step=40)

        teks = f"Rectified preview {n} | {item['nama_file']} | common corners: {item['jumlah_common']}"
        cv2.rectangle(gabung, (0, 0), (gabung.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(gabung, teks, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        path_out = os.path.join(PREVIEW_RECTIFIED_DIR, f"rectified_preview_{n:02d}_{item['nama_file']}")
        cv2.imwrite(path_out, gabung)
        output_paths.append(path_out)

    print(f"[OK] Preview rektifikasi tersimpan di: {PREVIEW_RECTIFIED_DIR}")
    return output_paths


# =========================================================
# VISUALISASI RINGKASAN STEREO
# =========================================================
def simpan_visualisasi_stereo(rms_stereo, baseline_m, baseline_mm, valid_count, image_size, R, T, E, F, Q):
    os.makedirs(VISUAL_STEREO_DIR, exist_ok=True)

    if not MATPLOTLIB_OK:
        print("[WARNING] matplotlib tidak tersedia. Visualisasi grafik stereo dilewati.")
        print("          Install dengan: python -m pip install matplotlib")
        return None

    path_dashboard = os.path.join(VISUAL_STEREO_DIR, "dashboard_kalibrasi_stereo.png")
    path_metric = os.path.join(VISUAL_STEREO_DIR, "metric_stereo_rms_baseline.png")

    # Grafik sederhana RMS stereo dan baseline.
    fig, ax1 = plt.subplots(figsize=(8, 6))
    labels = ["RMS stereo (px)", "Baseline (cm)"]
    values = [float(rms_stereo), float(baseline_mm / 10.0)]
    bars = ax1.bar(labels, values)
    ax1.set_title("Metrik Utama Kalibrasi Stereo")
    ax1.set_ylabel("Nilai")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path_metric, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Dashboard teks teknis.
    fig = plt.figure(figsize=(13, 9))
    plt.axis("off")
    teks = (
        "DASHBOARD KALIBRASI EKSTRINSIK / STEREO CHARUCO\n\n"
        f"Status umum\n"
        f"- RMS stereo        : {rms_stereo:.6f} px\n"
        f"- Baseline          : {baseline_m:.6f} m | {baseline_mm:.3f} mm | {baseline_mm/10:.3f} cm\n"
        f"- Pasangan valid    : {valid_count}\n"
        f"- Image size        : {image_size}\n\n"
        f"Translation Vector T\n{np.array2string(T.ravel(), precision=8)}\n\n"
        f"Rotation Matrix R\n{np.array2string(R, precision=8)}\n\n"
        f"Matriks Q dipakai untuk konversi disparity ke depth pada tahap berikutnya.\n"
        f"File utama yang dipakai aplikasi real-time adalah stereomap.xml.\n"
    )
    plt.text(0.02, 0.98, teks, va="top", ha="left", family="monospace", fontsize=11)
    fig.tight_layout()
    fig.savefig(path_dashboard, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Visualisasi stereo tersimpan: {VISUAL_STEREO_DIR}")
    return {
        "dashboard": path_dashboard,
        "metric": path_metric,
    }


# =========================================================
# MAIN
# =========================================================
def main():
    cek_aruco()

    print("\n" + "=" * 72)
    print("SCRIPT 3 - KALIBRASI EKSTRINSIK/STEREO CHARUCO + REKTIFIKASI")
    print("SELARAS DENGAN SCRIPT 2 + SCRIPT 06 CROP MULTI DETECT")
    print("=" * 72)
    print(f"Board              : {SQUARES_X} x {SQUARES_Y}")
    print(f"Square length      : {SQUARE_LENGTH} m / {SQUARE_LENGTH * 1000:.0f} mm")
    print(f"Marker length      : {MARKER_LENGTH} m / {MARKER_LENGTH * 1000:.0f} mm")
    print(f"Dictionary         : {ARUCO_DICT_NAME}")
    print(f"Folder kiri        : {FOLDER_LEFT}")
    print(f"Folder kanan       : {FOLDER_RIGHT}")
    print(f"Input kiri XML     : {INPUT_INTRINSIK_KIRI}")
    print(f"Input kanan XML    : {INPUT_INTRINSIK_KANAN}")
    print(f"Output stereomap   : {OUTPUT_STEREOMAP_XML}")
    print("-" * 72)
    print("PENGATURAN SELARAS SCRIPT 06")
    print(f"Input sudah crop dari Script 06 : {INPUT_SUDAH_CROP_DARI_SCRIPT_06}")
    print(f"Terapkan crop ulang            : {TERAPKAN_CROP_ULANG}")
    print(f"ROI kiri                       : {ROI_LEFT}")
    print(f"ROI kanan                      : {ROI_RIGHT}")
    print(f"CLAHE                          : {GUNAKAN_CLAHE}")
    print(f"Sharpen ringan                 : {GUNAKAN_SHARPEN_RINGAN}")
    print(f"Multi detect                   : {GUNAKAN_MULTI_DETECT}")
    print(f"Left boost                     : {LEFT_EXTRA_BOOST}")
    print("=" * 72)

    # 1. Muat intrinsik dari XML.
    print("\n[1] Memuat parameter intrinsik dari XML...")
    intrinsik_kiri = muat_intrinsik_xml("kiri", INPUT_INTRINSIK_KIRI)
    intrinsik_kanan = muat_intrinsik_xml("kanan", INPUT_INTRINSIK_KANAN)

    if intrinsik_kiri["image_size"] != intrinsik_kanan["image_size"]:
        raise RuntimeError(
            f"Image size intrinsik kiri {intrinsik_kiri['image_size']} tidak sama dengan kanan {intrinsik_kanan['image_size']}."
        )

    image_size = intrinsik_kiri["image_size"]

    pola_kiri = str(intrinsik_kiri["pola_terpilih"]).upper()
    pola_kanan = str(intrinsik_kanan["pola_terpilih"]).upper()

    if pola_kiri != pola_kanan:
        print(f"[WARNING] Pola intrinsik kiri ({pola_kiri}) dan kanan ({pola_kanan}) berbeda.")
        print("          Script memakai pola kamera kiri sebagai acuan.")

    legacy = pola_kiri == "LEGACY"
    board, aruco_dict = buat_charuco_board(legacy=legacy)
    print(f"[INFO] Pola ChArUco untuk stereo: {'LEGACY' if legacy else 'DEFAULT'}")

    # 2. Kumpulkan titik stereo ChArUco yang sama.
    objpoints, imgpoints_kiri, imgpoints_kanan, image_size_gambar, valid_pairs = kumpulkan_stereo_points_charuco(
        board,
        aruco_dict,
        image_size
    )

    valid_count = len(valid_pairs)

    if valid_count < 5:
        print(f"\n[STOP] Hanya {valid_count} pasangan gambar valid. Tidak cukup untuk stereo calibration.")
        print("       Idealnya gunakan minimal 15 sampai 30 pasangan valid dengan variasi sudut.")
        return

    if valid_count < 15:
        print("\n[WARNING] Pasangan valid kurang dari 15. Hasil bisa berjalan, tetapi stabilitas belum ideal.")
        print("          Disarankan menambah pasangan gambar dengan variasi sudut yang lebih banyak.")

    # 3. Stereo calibration.
    print("\n[3] Menghitung kalibrasi ekstrinsik/stereo...")
    print("    Intrinsik dikunci memakai cv2.CALIB_FIX_INTRINSIC")

    cameraMatrixL = intrinsik_kiri["camera_matrix"]
    distL = intrinsik_kiri["dist_coeffs"]
    cameraMatrixR = intrinsik_kanan["camera_matrix"]
    distR = intrinsik_kanan["dist_coeffs"]

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    rms_stereo, cML, dL, cMR, dR, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_kiri,
        imgpoints_kanan,
        cameraMatrixL,
        distL,
        cameraMatrixR,
        distR,
        image_size,
        criteria=criteria,
        flags=flags
    )

    baseline_m = float(np.linalg.norm(T))
    baseline_mm = baseline_m * 1000.0

    print("\n[HASIL STEREO CALIBRATION]")
    print(f"RMS stereo error : {rms_stereo:.6f} px")
    print(f"Baseline         : {baseline_m:.6f} m = {baseline_mm:.3f} mm = {baseline_mm/10:.3f} cm")
    print(f"Rotation Matrix R:\n{R}")
    print(f"Translation Vector T (meter):\n{T.ravel()}")
    print(f"Essential Matrix E:\n{E}")
    print(f"Fundamental Matrix F:\n{F}")

    if rms_stereo < 1.0:
        print("\nStatus stereo: BAGUS (RMS stereo < 1.0 px)")
    else:
        print("\nStatus stereo: PERLU DICEK (RMS stereo >= 1.0 px)")
        print("Saran: tambah pasangan gambar valid dengan variasi sudut dan posisi yang lebih baik.")

    # 4. Stereo rectification.
    print("\n[4] Menghitung stereo rectification...")
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

    # 5. Buat stereo maps.
    print("\n[5] Membuat stereo maps untuk cv2.remap...")
    stereoMapL_x, stereoMapL_y = cv2.initUndistortRectifyMap(
        cameraMatrixL,
        distL,
        R1,
        P1,
        image_size,
        cv2.CV_16SC2
    )

    stereoMapR_x, stereoMapR_y = cv2.initUndistortRectifyMap(
        cameraMatrixR,
        distR,
        R2,
        P2,
        image_size,
        cv2.CV_16SC2
    )

    # 6. Simpan XML.
    print("\n[6] Menyimpan hasil stereo ke XML...")
    simpan_stereomap_xml(
        OUTPUT_STEREOMAP_XML,
        stereoMapL_x,
        stereoMapL_y,
        stereoMapR_x,
        stereoMapR_y,
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
        baseline_m,
        baseline_mm,
        rms_stereo,
        image_size,
        valid_count,
        intrinsik_kiri,
        intrinsik_kanan
    )

    simpan_stereo_params_xml(
        OUTPUT_STEREO_PARAMS_XML,
        Q,
        R,
        T,
        E,
        F,
        R1,
        R2,
        P1,
        P2,
        baseline_m,
        baseline_mm,
        rms_stereo,
        image_size,
        valid_count
    )

    # 7. Simpan preview dan visualisasi.
    print("\n[7] Menyimpan preview rektifikasi dan visualisasi...")
    preview_paths = simpan_preview_rektifikasi(
        valid_pairs,
        stereoMapL_x,
        stereoMapL_y,
        stereoMapR_x,
        stereoMapR_y,
        max_preview=8
    )

    viz_paths = simpan_visualisasi_stereo(
        rms_stereo,
        baseline_m,
        baseline_mm,
        valid_count,
        image_size,
        R,
        T,
        E,
        F,
        Q
    )

    # Ringkasan akhir.
    print("\n" + "=" * 72)
    print("KALIBRASI EKSTRINSIK / STEREO SELESAI")
    print("=" * 72)
    print(f"Pasangan valid  : {valid_count}")
    print(f"RMS stereo      : {rms_stereo:.6f} px")
    print(f"Baseline        : {baseline_m:.6f} m | {baseline_mm:.3f} mm | {baseline_mm/10:.3f} cm")
    print(f"Stereomap XML   : {OUTPUT_STEREOMAP_XML}")
    print(f"Stereo params   : {OUTPUT_STEREO_PARAMS_XML}")
    print(f"Preview folder  : {PREVIEW_RECTIFIED_DIR}")
    print(f"Visual folder   : {VISUAL_STEREO_DIR}")
    print("\nLANGKAH BERIKUTNYA:")
    print("1. Buka preview_stereo_rectified dan cek apakah garis horizontal sudah sejajar.")
    print("2. Jika sejajar, stereomap.xml siap dipakai pada aplikasi real-time dengan cv2.remap.")
    print("3. Jika belum sejajar, tambah gambar kalibrasi dengan variasi posisi/sudut yang lebih baik.")
    print("=" * 72)


if __name__ == "__main__":
    main()
