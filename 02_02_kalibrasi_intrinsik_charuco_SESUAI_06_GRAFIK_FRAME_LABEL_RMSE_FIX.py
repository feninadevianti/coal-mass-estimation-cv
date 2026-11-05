
import cv2
from cv2 import aruco
import numpy as np
import glob
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError(
        "Modul matplotlib belum tersedia. Install dulu dengan:\n"
        "pip install matplotlib\n"
        f"Detail error: {e}"
    )


# =========================================================
# KONFIGURASI CHARUCO BOARD - SESUAI SCRIPT CAPTURE KAMU
# =========================================================
SQUARES_X = 5
SQUARES_Y = 7

SQUARE_LENGTH = 0.039   # meter, sama dengan 39 mm
MARKER_LENGTH = 0.029   # meter, sama dengan 29 mm

MIN_CHARUCO_CORNERS = 6
ARUCO_DICT_ID = aruco.DICT_4X4_50
ARUCO_DICT_NAME = "DICT_4X4_50"


# =========================================================
# KONFIGURASI SELARAS SCRIPT 06 CAPTURE
# =========================================================
WIDTH = 640
HEIGHT = 480

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
# KONFIGURASI FOLDER - SESUAI SCRIPT CAPTURE KAMU
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FOLDER_LEFT = os.path.join(BASE_DIR, "charuco_left")
FOLDER_RIGHT = os.path.join(BASE_DIR, "charuco_right")

OUTPUT_DIR = os.path.join(BASE_DIR, "hasil_kalibrasi")
PREVIEW_DIR = os.path.join(OUTPUT_DIR, "preview_charuco")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualisasi")


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

    if not hasattr(aruco, "calibrateCameraCharuco"):
        raise RuntimeError(
            "Fungsi aruco.calibrateCameraCharuco() tidak ditemukan di OpenCV kamu.\n"
            "Solusi: gunakan versi OpenCV yang menyediakan binding Python untuk fungsi ini.\n"
            "Cek cepat di terminal:\n"
            "python -c \"import cv2; from cv2 import aruco; print(cv2.__version__); print(hasattr(aruco, 'calibrateCameraCharuco'))\""
        )


# =========================================================
# MEMBUAT CHARUCO BOARD DEFAULT DAN LEGACY
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


# =========================================================
# DETEKTOR CHARUCO
# =========================================================
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

    marker_corners, marker_ids, rejected = aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    if marker_ids is None or len(marker_ids) == 0:
        return None, None, None, None

    retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
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
# FILE GAMBAR
# =========================================================
def ambil_daftar_gambar(folder):
    pola = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    daftar = []
    for p in pola:
        daftar.extend(glob.glob(os.path.join(folder, p)))
    return sorted(daftar)


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


def persiapkan_gambar_untuk_kalibrasi(img, nama_kamera):
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
# EVALUASI POLA DEFAULT VS LEGACY SECARA GLOBAL
# =========================================================
def evaluasi_folder(folder, nama_kamera, board, aruco_dict):
    daftar_gambar = ambil_daftar_gambar(folder)
    total_valid = 0
    total_corners = 0
    total_markers = 0

    for path in daftar_gambar:
        img = cv2.imread(path)
        if img is None:
            continue

        img_proses = persiapkan_gambar_untuk_kalibrasi(img, nama_kamera)
        gray = cv2.cvtColor(img_proses, cv2.COLOR_BGR2GRAY)
        hasil = detect_charuco_multi(gray, nama_kamera, board, aruco_dict)

        total_markers += hasil["jumlah_marker"]
        if hasil["jumlah_charuco"] >= MIN_CHARUCO_CORNERS:
            total_valid += 1
            total_corners += hasil["jumlah_charuco"]

    return total_valid, total_corners, total_markers


def pilih_pola_global():
    board_default, dict_default = buat_charuco_board(legacy=False)
    board_legacy, dict_legacy = buat_charuco_board(legacy=True)

    kiri_def = evaluasi_folder(FOLDER_LEFT, "kiri", board_default, dict_default)
    kanan_def = evaluasi_folder(FOLDER_RIGHT, "kanan", board_default, dict_default)
    kiri_leg = evaluasi_folder(FOLDER_LEFT, "kiri", board_legacy, dict_legacy)
    kanan_leg = evaluasi_folder(FOLDER_RIGHT, "kanan", board_legacy, dict_legacy)

    skor_default = ((kiri_def[0] + kanan_def[0]) * 1000000
                    + (kiri_def[1] + kanan_def[1]) * 1000
                    + (kiri_def[2] + kanan_def[2]))

    skor_legacy = ((kiri_leg[0] + kanan_leg[0]) * 1000000
                   + (kiri_leg[1] + kanan_leg[1]) * 1000
                   + (kiri_leg[2] + kanan_leg[2]))

    print("\n" + "=" * 70)
    print("EVALUASI POLA CHARUCO")
    print("=" * 70)
    print("DEFAULT:")
    print(f"  Kiri  - valid: {kiri_def[0]}, corners: {kiri_def[1]}, markers: {kiri_def[2]}")
    print(f"  Kanan - valid: {kanan_def[0]}, corners: {kanan_def[1]}, markers: {kanan_def[2]}")
    print("LEGACY:")
    print(f"  Kiri  - valid: {kiri_leg[0]}, corners: {kiri_leg[1]}, markers: {kiri_leg[2]}")
    print(f"  Kanan - valid: {kanan_leg[0]}, corners: {kanan_leg[1]}, markers: {kanan_leg[2]}")

    if skor_legacy > skor_default:
        print("\n[INFO] Pola yang dipilih: LEGACY")
        print("=" * 70)
        return "LEGACY", board_legacy, dict_legacy

    print("\n[INFO] Pola yang dipilih: DEFAULT")
    print("=" * 70)
    return "DEFAULT", board_default, dict_default


# =========================================================
# ERROR REPROJECTION
# =========================================================
def hitung_reprojection_error(board, all_corners, all_ids, rvecs, tvecs, camera_matrix, dist_coeffs):
    """
    Menghitung reprojection error dengan rumus RMSE per frame.

    Catatan penting:
    Cara lama cv2.norm(...) / jumlah_titik dapat terlihat terlalu kecil.
    Untuk dibandingkan dengan RMS dari calibrateCameraCharuco, rumus yang lebih konsisten adalah:
        RMSE = sqrt(mean(error_per_titik^2))
    """
    semua_objek_corner = np.asarray(board.getChessboardCorners(), dtype=np.float32)         if hasattr(board, "getChessboardCorners") else np.asarray(board.chessboardCorners, dtype=np.float32)

    total_squared_error = 0.0
    total_points = 0
    per_frame_error = []

    for i in range(len(all_corners)):
        ids = all_ids[i].flatten().astype(int)
        imgpoints = all_corners[i].astype(np.float32)
        objpoints = semua_objek_corner[ids].reshape(-1, 1, 3).astype(np.float32)

        imgpoints_reproj, _ = cv2.projectPoints(
            objpoints,
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )

        titik_asli = imgpoints.reshape(-1, 2).astype(np.float64)
        titik_reproj = imgpoints_reproj.reshape(-1, 2).astype(np.float64)
        diff = titik_asli - titik_reproj

        squared_per_point = np.sum(diff * diff, axis=1)
        frame_rmse = np.sqrt(np.mean(squared_per_point))

        per_frame_error.append(float(frame_rmse))
        total_squared_error += float(np.sum(squared_per_point))
        total_points += len(squared_per_point)

    overall_rmse_manual = np.sqrt(total_squared_error / max(total_points, 1))
    return float(overall_rmse_manual), per_frame_error


# =========================================================
# PREVIEW
# =========================================================
def simpan_preview(path_preview, img, hasil, status_text):
    vis = img.copy()

    if hasil["marker_corners"] is not None and hasil["marker_ids"] is not None:
        try:
            aruco.drawDetectedMarkers(vis, hasil["marker_corners"], hasil["marker_ids"])
        except Exception:
            pass

    if hasil["charuco_corners"] is not None and hasil["charuco_ids"] is not None:
        try:
            aruco.drawDetectedCornersCharuco(vis, hasil["charuco_corners"], hasil["charuco_ids"])
        except Exception:
            titik = hasil["charuco_corners"].reshape(-1, 2)
            for p in titik:
                cv2.circle(vis, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)

    warna = (0, 255, 0) if status_text.startswith("VALID") else (0, 0, 255)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(vis, status_text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, warna, 2)

    cv2.imwrite(path_preview, vis)


# =========================================================
# KALIBRASI SATU KAMERA
# =========================================================
def kalibrasi_intrinsik_satu_kamera(nama_kamera, folder, board, aruco_dict, pola_terpilih):
    print("\n" + "-" * 70)
    print(f"MEMPROSES KAMERA {nama_kamera.upper()}")
    print(f"Folder: {folder}")
    print(f"Pola  : {pola_terpilih}")
    print("-" * 70)

    daftar_gambar = ambil_daftar_gambar(folder)
    if len(daftar_gambar) == 0:
        print(f"[ERROR] Tidak ada gambar di folder: {folder}")
        return None

    preview_folder = os.path.join(PREVIEW_DIR, nama_kamera)
    os.makedirs(preview_folder, exist_ok=True)

    all_corners = []
    all_ids = []
    valid_files = []
    image_size = None

    for idx, path in enumerate(daftar_gambar, 1):
        nama_file = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"[{idx:02d}] GAGAL BACA: {nama_file}")
            continue

        img_proses = persiapkan_gambar_untuk_kalibrasi(img, nama_kamera)
        gray = cv2.cvtColor(img_proses, cv2.COLOR_BGR2GRAY)
        sharp = hitung_sharpness(gray)
        ukuran_ini = (gray.shape[1], gray.shape[0])

        if image_size is None:
            image_size = ukuran_ini
        elif ukuran_ini != image_size:
            print(f"[{idx:02d}] DILEWATI: {nama_file} - ukuran {ukuran_ini}, harus {image_size}")
            continue

        hasil = detect_charuco_multi(gray, nama_kamera, board, aruco_dict)
        jumlah = hasil["jumlah_charuco"]
        marker = hasil["jumlah_marker"]
        variasi = hasil.get("variasi", "-")

        if jumlah >= MIN_CHARUCO_CORNERS:
            all_corners.append(hasil["charuco_corners"])
            all_ids.append(hasil["charuco_ids"])
            valid_files.append(nama_file)
            print(f"[{idx:02d}] VALID   : {nama_file} - corners: {jumlah}, markers: {marker}, deteksi: {variasi}, sharp: {sharp:.1f}")
            status = f"VALID - corners: {jumlah}, markers: {marker}, {pola_terpilih}, {variasi}, sharp: {sharp:.1f}"
        else:
            print(f"[{idx:02d}] DILEWATI: {nama_file} - corners: {jumlah}, markers: {marker}, deteksi: {variasi}, sharp: {sharp:.1f}")
            status = f"TIDAK VALID - corners: {jumlah}, markers: {marker}, {pola_terpilih}, {variasi}, sharp: {sharp:.1f}"

        simpan_preview(os.path.join(preview_folder, nama_file), img_proses, hasil, status)

    n_valid = len(all_corners)

    print("\n" + "-" * 70)
    print(f"Gambar valid {nama_kamera}: {n_valid} dari {len(daftar_gambar)}")
    print("-" * 70)

    if n_valid == 0:
        print("[STOP] Tidak ada gambar valid.")
        print("Cek board: SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, dan DICT_4X4_50.")
        return None

    if n_valid < 5:
        print("[WARNING] Gambar valid kurang dari 5. Hasil kalibrasi kemungkinan tidak stabil.")
    if n_valid < 15:
        print("[SARAN] Idealnya gunakan 15 sampai 30 gambar valid dengan variasi sudut.")

    print("\nMenghitung kalibrasi intrinsik dengan aruco.calibrateCameraCharuco...")

    try:
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
            all_corners,
            all_ids,
            board,
            image_size,
            None,
            None
        )
    except cv2.error as e:
        print("[ERROR] aruco.calibrateCameraCharuco gagal.")
        print(e)
        return None

    mean_error, per_frame_error = hitung_reprojection_error(
        board,
        all_corners,
        all_ids,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs
    )

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        1,
        image_size
    )

    if mean_error < 0.5:
        status = "BAGUS"
    elif mean_error < 1.0:
        status = "CUKUP"
    else:
        status = "KURANG"

    print("\n" + "=" * 70)
    print(f"HASIL INTRINSIK CHARUCO - {nama_kamera.upper()}")
    print("=" * 70)
    print(f"RMS Error               : {rms:.6f} px")
    print(f"Manual Overall RMSE     : {mean_error:.6f} px")
    print(f"Status                  : {status}")
    print(f"Gambar valid            : {n_valid}")
    print(f"Image size              : {image_size}")
    print("\nCamera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs.ravel())
    print("\nError per frame:")
    for i, err in enumerate(per_frame_error, 1):
        print(f"Frame {i:02d} | {valid_files[i - 1]} | {err:.6f} px")
    print("=" * 70)

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "new_camera_matrix": new_camera_matrix,
        "roi": roi,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "rms": float(rms),
        "mean_error": float(mean_error),
        "image_size": image_size,
        "n_valid_frames": n_valid,
        "per_frame_error": np.array(per_frame_error, dtype=np.float32),
        "valid_files": np.array(valid_files),
        "pola_terpilih": pola_terpilih,
        "status": status,
        "crop_aktif": bool(CROP_AKTIF),
        "simpan_frame_crop": bool(SIMPAN_FRAME_CROP),
        "input_sudah_crop_dari_script_06": bool(INPUT_SUDAH_CROP_DARI_SCRIPT_06),
        "terapkan_crop_ulang": bool(TERAPKAN_CROP_ULANG),
        "roi_capture": np.array(pilih_roi(nama_kamera), dtype=np.float32),
        "gunakan_clahe": bool(GUNAKAN_CLAHE),
        "gunakan_sharpen_ringan": bool(GUNAKAN_SHARPEN_RINGAN),
        "gunakan_multi_detect": bool(GUNAKAN_MULTI_DETECT),
        "left_extra_boost": bool(LEFT_EXTRA_BOOST),
    }


# =========================================================
# SIMPAN HASIL XML
# =========================================================
def tulis_list_string(fs, key, values):
    fs.startWriteStruct(key, cv2.FileNode_SEQ)
    for v in values:
        fs.write("", str(v))
    fs.endWriteStruct()


def simpan_hasil_xml(nama_kamera, hasil):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path_xml = os.path.join(OUTPUT_DIR, f"intrinsik_{nama_kamera}.xml")

    fs = cv2.FileStorage(path_xml, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Gagal membuka file XML untuk ditulis: {path_xml}")

    fs.write("camera_matrix", hasil["camera_matrix"])
    fs.write("dist_coeffs", hasil["dist_coeffs"])
    fs.write("new_camera_matrix", hasil["new_camera_matrix"])
    fs.write("roi", np.array(hasil["roi"], dtype=np.int32))
    fs.write("rms", float(hasil["rms"]))
    fs.write("mean_error", float(hasil["mean_error"]))
    fs.write("image_width", int(hasil["image_size"][0]))
    fs.write("image_height", int(hasil["image_size"][1]))
    fs.write("n_valid_frames", int(hasil["n_valid_frames"]))
    fs.write("per_frame_error", hasil["per_frame_error"].reshape(-1, 1))
    fs.write("pola_terpilih", str(hasil["pola_terpilih"]))
    fs.write("status", str(hasil["status"]))
    fs.write("squares_x", int(SQUARES_X))
    fs.write("squares_y", int(SQUARES_Y))
    fs.write("square_length", float(SQUARE_LENGTH))
    fs.write("marker_length", float(MARKER_LENGTH))
    fs.write("aruco_dict_name", str(ARUCO_DICT_NAME))
    fs.write("capture_script_reference", "06_CAPTURE_CHARUCO_STEREO_LEFT_BOOST_MULTI_DETECT")
    fs.write("crop_aktif", int(hasil.get("crop_aktif", CROP_AKTIF)))
    fs.write("simpan_frame_crop", int(hasil.get("simpan_frame_crop", SIMPAN_FRAME_CROP)))
    fs.write("input_sudah_crop_dari_script_06", int(hasil.get("input_sudah_crop_dari_script_06", INPUT_SUDAH_CROP_DARI_SCRIPT_06)))
    fs.write("terapkan_crop_ulang", int(hasil.get("terapkan_crop_ulang", TERAPKAN_CROP_ULANG)))
    fs.write("roi_capture", hasil.get("roi_capture", np.array(ROI_LEFT, dtype=np.float32)).reshape(1, 4))
    fs.write("gunakan_clahe", int(hasil.get("gunakan_clahe", GUNAKAN_CLAHE)))
    fs.write("gunakan_sharpen_ringan", int(hasil.get("gunakan_sharpen_ringan", GUNAKAN_SHARPEN_RINGAN)))
    fs.write("gunakan_multi_detect", int(hasil.get("gunakan_multi_detect", GUNAKAN_MULTI_DETECT)))
    fs.write("left_extra_boost", int(hasil.get("left_extra_boost", LEFT_EXTRA_BOOST)))
    fs.write("camera_height_cm", float(CAMERA_HEIGHT_CM))
    fs.write("camera_tilt_min_deg", float(CAMERA_TILT_MIN_DEG))
    fs.write("camera_tilt_max_deg", float(CAMERA_TILT_MAX_DEG))
    fs.write("camera_fov_deg", float(CAMERA_FOV_DEG))
    fs.write("gerbong_panjang_cm", float(GERBONG_PANJANG_CM))
    fs.write("gerbong_lebar_cm", float(GERBONG_LEBAR_CM))
    fs.write("gerbong_tinggi_cm", float(GERBONG_TINGGI_CM))
    tulis_list_string(fs, "valid_files", hasil["valid_files"])
    fs.release()

    print(f"[OK] Hasil tersimpan XML: {path_xml}")
    return path_xml


# =========================================================
# FUNGSI VISUALISASI / GRAFIK
# =========================================================
def array_to_pretty_string(arr, precision=6):
    return np.array2string(np.asarray(arr), precision=precision, suppress_small=False)


def simpan_grafik_batang_dua_kamera(nilai_kiri, nilai_kanan, judul, ylabel, path_out):
    plt.figure(figsize=(8, 6))
    labels = ["Kiri", "Kanan"]
    values = [nilai_kiri, nilai_kanan]
    bars = plt.bar(labels, values)
    plt.title(judul)
    plt.ylabel(ylabel)
    plt.xlabel("Kamera")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.6f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(path_out, dpi=200, bbox_inches="tight")
    plt.close()


def label_frame_dari_nama_file(nama_file):
    """Mengubah nama file menjadi label ringkas di bawah batang grafik."""
    base = os.path.basename(str(nama_file))
    label, _ = os.path.splitext(base)
    return label


def warna_error_reprojection(err):
    if err < 0.5:
        return "#2ecc71"  # GOOD, hijau
    if err < 1.0:
        return "#f39c12"  # OK, oranye
    return "#e74c3c"      # HIGH, merah


def simpan_grafik_error_per_frame(per_frame_error, valid_files, nama_kamera, rms_value, path_out):
    y = np.asarray(per_frame_error, dtype=float)
    n = len(y)

    if n == 0:
        print(f"[WARNING] Tidak ada error per frame untuk kamera {nama_kamera}.")
        return

    labels = [label_frame_dari_nama_file(v) for v in valid_files]
    if len(labels) != n:
        labels = [f"frame_{i:03d}" for i in range(1, n + 1)]

    x = np.arange(n)
    warna = [warna_error_reprojection(err) for err in y]

    lebar_fig = max(12, n * 0.45)
    plt.figure(figsize=(lebar_fig, 6.8))

    plt.bar(x, y, color=warna, width=0.75)
    plt.axhline(float(rms_value), linestyle="-.", linewidth=1.5, color="#3498db")

    y_max = max(float(np.max(y)) * 1.25, float(rms_value) * 1.35, 0.10)
    plt.ylim(0, y_max)

    plt.title(
        f"Per-Image Reprojection Error - ChArUco Calibration - Kamera {nama_kamera.upper()}",
        fontsize=13,
        fontweight="bold"
    )
    plt.xlabel("Nama Frame/Gambar")
    plt.ylabel("Reprojection Error (px)")
    plt.xticks(x, labels, rotation=90, fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_items = [
        Patch(facecolor="#2ecc71", label="GOOD (< 0.50 px)"),
        Patch(facecolor="#f39c12", label="OK (0.50–1.00 px)"),
        Patch(facecolor="#e74c3c", label="HIGH (> 1.00 px)"),
        Line2D([0], [0], color="#3498db", linestyle="-.", lw=1.5,
               label=f"Overall RMS = {float(rms_value):.3f} px")
    ]
    plt.legend(handles=legend_items, loc="upper right", frameon=True)

    # Tulis nilai error kecil di atas batang jika jumlah frame tidak terlalu padat.
    if n <= 40:
        for xi, yi in zip(x, y):
            plt.text(xi, yi + (y_max * 0.015), f"{yi:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(path_out, dpi=220, bbox_inches="tight")
    plt.close()


def simpan_ringkasan_kamera_gambar(nama_kamera, hasil, path_out):
    fig = plt.figure(figsize=(10, 7))
    plt.axis("off")

    matrix_str = array_to_pretty_string(hasil["camera_matrix"], precision=6)
    dist_str = array_to_pretty_string(hasil["dist_coeffs"].ravel(), precision=6)
    roi_str = str(tuple(int(x) for x in hasil["roi"]))

    teks = (
        f"RINGKASAN KALIBRASI INTRINSIK - KAMERA {nama_kamera.upper()}\n\n"
        f"Status                  : {hasil['status']}\n"
        f"Pola terpilih           : {hasil['pola_terpilih']}\n"
        f"Gambar valid            : {hasil['n_valid_frames']}\n"
        f"Image size              : {hasil['image_size']}\n"
        f"RMS Error               : {hasil['rms']:.6f} px\n"
        f"Manual Overall RMSE     : {hasil['mean_error']:.6f} px\n"
        f"ROI                     : {roi_str}\n\n"
        f"Camera Matrix:\n{matrix_str}\n\n"
        f"Distortion Coefficients:\n{dist_str}\n"
    )

    plt.text(0.02, 0.98, teks, va="top", ha="left", family="monospace", fontsize=11)
    plt.tight_layout()
    plt.savefig(path_out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def simpan_dashboard_ringkasan(hasil_kiri, hasil_kanan, path_out):
    fig = plt.figure(figsize=(12, 8))
    plt.axis("off")

    kiri_cm = hasil_kiri["camera_matrix"]
    kanan_cm = hasil_kanan["camera_matrix"]

    teks = (
        "DASHBOARD RINGKASAN KALIBRASI INTRINSIK STEREO\n\n"
        f"KAMERA KIRI\n"
        f"  Status                  : {hasil_kiri['status']}\n"
        f"  Gambar valid            : {hasil_kiri['n_valid_frames']}\n"
        f"  RMS Error               : {hasil_kiri['rms']:.6f} px\n"
        f"  Mean Reprojection Error : {hasil_kiri['mean_error']:.6f} px\n"
        f"  fx, fy                  : {kiri_cm[0,0]:.6f}, {kiri_cm[1,1]:.6f}\n"
        f"  cx, cy                  : {kiri_cm[0,2]:.6f}, {kiri_cm[1,2]:.6f}\n\n"
        f"KAMERA KANAN\n"
        f"  Status                  : {hasil_kanan['status']}\n"
        f"  Gambar valid            : {hasil_kanan['n_valid_frames']}\n"
        f"  RMS Error               : {hasil_kanan['rms']:.6f} px\n"
        f"  Mean Reprojection Error : {hasil_kanan['mean_error']:.6f} px\n"
        f"  fx, fy                  : {kanan_cm[0,0]:.6f}, {kanan_cm[1,1]:.6f}\n"
        f"  cx, cy                  : {kanan_cm[0,2]:.6f}, {kanan_cm[1,2]:.6f}\n\n"
        "Interpretasi singkat:\n"
        "- Mean reprojection error yang semakin kecil menunjukkan kalibrasi semakin baik.\n"
        "- Camera matrix memuat focal length (fx, fy) dan optical center (cx, cy).\n"
        "- Distortion coefficients dipakai untuk koreksi distorsi lensa.\n"
        "- Jika hasil kiri dan kanan sudah baik, proses dapat dilanjutkan ke stereo calibration.\n"
    )

    plt.text(0.02, 0.98, teks, va="top", ha="left", family="monospace", fontsize=12)
    plt.tight_layout()
    plt.savefig(path_out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def simpan_grafik_focal_length(hasil_kiri, hasil_kanan, path_out):
    kiri_cm = hasil_kiri["camera_matrix"]
    kanan_cm = hasil_kanan["camera_matrix"]

    labels = ["Kiri fx", "Kiri fy", "Kanan fx", "Kanan fy"]
    values = [kiri_cm[0, 0], kiri_cm[1, 1], kanan_cm[0, 0], kanan_cm[1, 1]]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, values)
    plt.title("Representasi Focal Length fx dan fy")
    plt.ylabel("Nilai focal length (px)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}",
                 ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(path_out, dpi=200, bbox_inches="tight")
    plt.close()


def simpan_grafik_principal_point(hasil_kiri, hasil_kanan, path_out):
    kiri_cm = hasil_kiri["camera_matrix"]
    kanan_cm = hasil_kanan["camera_matrix"]

    labels = ["Kiri cx", "Kiri cy", "Kanan cx", "Kanan cy"]
    values = [kiri_cm[0, 2], kiri_cm[1, 2], kanan_cm[0, 2], kanan_cm[1, 2]]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, values)
    plt.title("Representasi Principal Point cx dan cy")
    plt.ylabel("Nilai principal point (px)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}",
                 ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(path_out, dpi=200, bbox_inches="tight")
    plt.close()


def simpan_grafik_distortion_coefficients(hasil_kiri, hasil_kanan, path_out):
    dist_kiri = hasil_kiri["dist_coeffs"].ravel()
    dist_kanan = hasil_kanan["dist_coeffs"].ravel()
    n = min(len(dist_kiri), len(dist_kanan))
    labels = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"][:n]

    x = np.arange(n)
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, dist_kiri[:n], width, label="Kiri")
    plt.bar(x + width / 2, dist_kanan[:n], width, label="Kanan")
    plt.axhline(0, linewidth=1)
    plt.title("Representasi Distortion Coefficients")
    plt.xlabel("Koefisien distorsi")
    plt.ylabel("Nilai koefisien")
    plt.xticks(x, labels)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_out, dpi=200, bbox_inches="tight")
    plt.close()


def simpan_semua_visualisasi(hasil_kiri, hasil_kanan):
    os.makedirs(VIZ_DIR, exist_ok=True)

    path_mean_bar = os.path.join(VIZ_DIR, "manual_overall_rmse_bar.png")
    path_rms_bar = os.path.join(VIZ_DIR, "rms_error_bar.png")
    path_per_frame_kiri = os.path.join(VIZ_DIR, "error_per_frame_kiri.png")
    path_per_frame_kanan = os.path.join(VIZ_DIR, "error_per_frame_kanan.png")
    path_ringkasan_kiri = os.path.join(VIZ_DIR, "ringkasan_intrinsik_kiri.png")
    path_ringkasan_kanan = os.path.join(VIZ_DIR, "ringkasan_intrinsik_kanan.png")
    path_focal = os.path.join(VIZ_DIR, "focal_length_fx_fy.png")
    path_principal = os.path.join(VIZ_DIR, "principal_point_cx_cy.png")
    path_distortion = os.path.join(VIZ_DIR, "distortion_coefficients.png")
    path_dashboard = os.path.join(VIZ_DIR, "dashboard_kalibrasi_intrinsik.png")

    simpan_grafik_batang_dua_kamera(
        hasil_kiri["mean_error"], hasil_kanan["mean_error"],
        "Perbandingan Manual Overall RMSE", "Error (px)", path_mean_bar
    )

    simpan_grafik_batang_dua_kamera(
        hasil_kiri["rms"], hasil_kanan["rms"],
        "Perbandingan RMS Error", "Error (px)", path_rms_bar
    )

    simpan_grafik_error_per_frame(
        hasil_kiri["per_frame_error"],
        hasil_kiri["valid_files"],
        "kiri",
        hasil_kiri["rms"],
        path_per_frame_kiri
    )
    simpan_grafik_error_per_frame(
        hasil_kanan["per_frame_error"],
        hasil_kanan["valid_files"],
        "kanan",
        hasil_kanan["rms"],
        path_per_frame_kanan
    )

    simpan_ringkasan_kamera_gambar("kiri", hasil_kiri, path_ringkasan_kiri)
    simpan_ringkasan_kamera_gambar("kanan", hasil_kanan, path_ringkasan_kanan)
    simpan_grafik_focal_length(hasil_kiri, hasil_kanan, path_focal)
    simpan_grafik_principal_point(hasil_kiri, hasil_kanan, path_principal)
    simpan_grafik_distortion_coefficients(hasil_kiri, hasil_kanan, path_distortion)
    simpan_dashboard_ringkasan(hasil_kiri, hasil_kanan, path_dashboard)

    print("[OK] Visualisasi tersimpan:")
    print(f"  - {path_mean_bar}")
    print(f"  - {path_rms_bar}")
    print(f"  - {path_per_frame_kiri}")
    print(f"  - {path_per_frame_kanan}")
    print(f"  - {path_ringkasan_kiri}")
    print(f"  - {path_ringkasan_kanan}")
    print(f"  - {path_focal}")
    print(f"  - {path_principal}")
    print(f"  - {path_distortion}")
    print(f"  - {path_dashboard}")

    return {
        "mean_bar": path_mean_bar,
        "rms_bar": path_rms_bar,
        "per_frame_kiri": path_per_frame_kiri,
        "per_frame_kanan": path_per_frame_kanan,
        "ringkasan_kiri": path_ringkasan_kiri,
        "ringkasan_kanan": path_ringkasan_kanan,
        "focal_length": path_focal,
        "principal_point": path_principal,
        "distortion_coefficients": path_distortion,
        "dashboard": path_dashboard,
    }


# =========================================================
# MAIN
# =========================================================
def main():
    cek_aruco()

    print("\n" + "=" * 70)
    print("SCRIPT 2 - KALIBRASI INTRINSIK CHARUCO - XML + GRAFIK")
    print("DISELARASKAN DENGAN SCRIPT 06: CROP + CLAHE + SHARPEN + MULTI DETECT")
    print("=" * 70)
    print(f"Board              : {SQUARES_X} x {SQUARES_Y}")
    print(f"Square length      : {SQUARE_LENGTH} m / {SQUARE_LENGTH * 1000:.0f} mm")
    print(f"Marker length      : {MARKER_LENGTH} m / {MARKER_LENGTH * 1000:.0f} mm")
    print(f"Dictionary         : {ARUCO_DICT_NAME}")
    print(f"Folder kiri        : {FOLDER_LEFT}")
    print(f"Folder kanan       : {FOLDER_RIGHT}")
    print(f"Output             : {OUTPUT_DIR}")
    print(f"Visualisasi        : {VIZ_DIR}")
    print("-" * 70)
    print("PENGATURAN SELARAS SCRIPT 06")
    print(f"Input sudah crop dari Script 06 : {INPUT_SUDAH_CROP_DARI_SCRIPT_06}")
    print(f"Terapkan crop ulang            : {TERAPKAN_CROP_ULANG}")
    print(f"ROI kiri                       : {ROI_LEFT}")
    print(f"ROI kanan                      : {ROI_RIGHT}")
    print(f"CLAHE                          : {GUNAKAN_CLAHE}")
    print(f"Sharpen ringan                 : {GUNAKAN_SHARPEN_RINGAN}")
    print(f"Multi detect                   : {GUNAKAN_MULTI_DETECT}")
    print(f"Left boost                     : {LEFT_EXTRA_BOOST}")
    print("=" * 70)

    pola_terpilih, board, aruco_dict = pilih_pola_global()

    hasil_kiri = kalibrasi_intrinsik_satu_kamera(
        "kiri",
        FOLDER_LEFT,
        board,
        aruco_dict,
        pola_terpilih
    )
    if hasil_kiri is None:
        print("\n[STOP] Kalibrasi kamera kiri gagal.")
        return

    hasil_kanan = kalibrasi_intrinsik_satu_kamera(
        "kanan",
        FOLDER_RIGHT,
        board,
        aruco_dict,
        pola_terpilih
    )
    if hasil_kanan is None:
        print("\n[STOP] Kalibrasi kamera kanan gagal.")
        return

    print("\nMenyimpan hasil kalibrasi XML...")
    path_kiri = simpan_hasil_xml("kiri", hasil_kiri)
    path_kanan = simpan_hasil_xml("kanan", hasil_kanan)

    print("\nMembuat grafik dan gambar ringkasan...")
    path_viz = simpan_semua_visualisasi(hasil_kiri, hasil_kanan)

    print("\n" + "=" * 70)
    print("RINGKASAN AKHIR")
    print("=" * 70)
    print(f"Pola terpilih : {pola_terpilih}")
    print(f"Kamera kiri   : valid {hasil_kiri['n_valid_frames']} | mean error {hasil_kiri['mean_error']:.6f} px")
    print(f"Kamera kanan  : valid {hasil_kanan['n_valid_frames']} | mean error {hasil_kanan['mean_error']:.6f} px")
    print(f"File kiri     : {path_kiri}")
    print(f"File kanan    : {path_kanan}")
    print(f"Grafik RMSE   : {path_viz['mean_bar']}")
    print(f"Dashboard     : {path_viz['dashboard']}")

    if hasil_kiri["mean_error"] < 1.0 and hasil_kanan["mean_error"] < 1.0:
        print("\n[OK] Kalibrasi intrinsik berhasil. Lanjut ke kalibrasi stereo.")
    else:
        print("\n[PERINGATAN] Error salah satu kamera >= 1.0 px.")
        print("Saran: tambah gambar valid dengan variasi sudut lebih baik.")
    print("=" * 70)


if __name__ == "__main__":
    main()
