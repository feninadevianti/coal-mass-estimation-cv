import cv2
from cv2 import aruco
import numpy as np
import os
import time

CAMERA_LEFT = 0
CAMERA_RIGHT = 1

CAMERA_ID_CANDIDATES = [0, 1, 2, 3, 4, 5]

WIDTH = 640
HEIGHT = 480
FPS = 30
DISPLAY_MIRROR_HORIZONTAL = True
CROP_AKTIF = True
SIMPAN_FRAME_CROP = True
ROI_LEFT = (0.00, 0.05, 1.00, 0.95)
ROI_RIGHT = (0.00, 0.05, 1.00, 0.95)

CAMERA_HEIGHT_CM = 60.0
CAMERA_TILT_MIN_DEG = 10.0
CAMERA_TILT_MAX_DEG = 15.0
CAMERA_FOV_DEG = 120.0

GERBONG_PANJANG_CM = 40.0
GERBONG_LEBAR_CM = 10.0
GERBONG_TINGGI_CM = 7.5

TARGET_BOARD_MIN = 0.30
TARGET_BOARD_MAX = 0.60

TAMPILKAN_KOTAK_PANDUAN = True

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
BLOKIR_SIMPAN_JIKA_BLUR = False

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

SQUARES_X = 5
SQUARES_Y = 7

SQUARE_LENGTH = 0.039   # 39 mm, hasil ukur satu kotak penuh
MARKER_LENGTH = 0.029   # 29 mm, ukur sisi marker ArUco kecil

MIN_CHARUCO_CORNERS = 6

# =========================================================
# FOLDER OUTPUT
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FOLDER_LEFT = os.path.join(BASE_DIR, "charuco_left")
FOLDER_RIGHT = os.path.join(BASE_DIR, "charuco_right")

os.makedirs(FOLDER_LEFT, exist_ok=True)
os.makedirs(FOLDER_RIGHT, exist_ok=True)

# =========================================================
# ARUCO DICTIONARY
# =========================================================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# =========================================================
# MEMBUAT CHARUCO DETECTOR DEFAULT DAN LEGACY
# =========================================================
def buat_charuco_detector(legacy=False):
    board = aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        aruco_dict
    )

    # Beberapa board dari generator seperti calib.io kadang cocoknya legacy pattern
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(legacy)

    detector = aruco.CharucoDetector(board)

    return detector


charuco_detector_default = buat_charuco_detector(legacy=False)
charuco_detector_legacy = buat_charuco_detector(legacy=True)

# =========================================================
# BACKEND KAMERA
# =========================================================
def daftar_backend():
    backends = []

    # CAP_ANY paling aman dicoba pertama untuk beberapa laptop
    backends.append((cv2.CAP_ANY, "CAP_ANY"))

    if os.name == "nt":
        backends.append((cv2.CAP_MSMF, "MSMF"))
        backends.append((cv2.CAP_DSHOW, "DSHOW"))

    return backends


def set_property_aman(cap, prop, value):
    try:
        cap.set(prop, value)
    except Exception:
        pass


def mirror_untuk_tampilan(img):
    return cv2.flip(img, 1) if DISPLAY_MIRROR_HORIZONTAL else img


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
        # Nilai 0.25 umum dipakai di DirectShow untuk manual exposure. Jika tidak cocok, kamera akan mengabaikan.
        set_property_aman(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        set_property_aman(cap, cv2.CAP_PROP_EXPOSURE, EXPOSURE_VALUE)

    if KUNCI_AUTOFOCUS and hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
        set_property_aman(cap, cv2.CAP_PROP_AUTOFOCUS, 0)
        if str(nama).upper() == "KIRI":
            set_property_aman(cap, cv2.CAP_PROP_FOCUS, FOCUS_LEFT)
        else:
            set_property_aman(cap, cv2.CAP_PROP_FOCUS, FOCUS_RIGHT)

    print(f"[INFO] Setting kamera {nama} dicoba disamakan: brightness={BRIGHTNESS_VALUE}, contrast={CONTRAST_VALUE}, gain={GAIN_VALUE}, WB={WB_TEMPERATURE}")


def hitung_sharpness(gray):
    if gray is None:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


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
        blur = cv2.GaussianBlur(gray, (0, 0), SHARPEN_SIGMA)
        gray = cv2.addWeighted(gray, 1.0 + SHARPEN_AMOUNT, blur, -SHARPEN_AMOUNT, 0)

    return gray


def sharpen_gray(gray, amount=0.55, sigma=1.0):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)


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
    variasi = []
    variasi.append(("RAW", gray))

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )
    gray_clahe = clahe.apply(gray)
    variasi.append(("CLAHE", gray_clahe))

    gray_sharp = sharpen_gray(gray_clahe, amount=SHARPEN_AMOUNT, sigma=SHARPEN_SIGMA)
    variasi.append(("CLAHE_SHARP", gray_sharp))

    gray_eq = cv2.equalizeHist(gray)
    variasi.append(("EQUALIZE", gray_eq))

    gray_contrast = cv2.convertScaleAbs(gray, alpha=1.25, beta=5)
    variasi.append(("CONTRAST", gray_contrast))

    if LEFT_EXTRA_BOOST and str(nama_kamera).upper() == "KIRI":
        variasi.append(("LEFT_BOOST", boost_left_gray(gray)))

    return variasi


def skor_hasil_deteksi(hasil):
    return (hasil["jumlah_charuco"] * 1000) + hasil["jumlah_marker"]


# =========================================================
# BUKA KAMERA
# =========================================================
def open_camera(cam_id, nama):
    for backend, backend_name in daftar_backend():
        print(f"[INFO] Membuka kamera {nama} ID {cam_id} dengan backend {backend_name}...")

        cap = cv2.VideoCapture(cam_id, backend)

        if not cap.isOpened():
            print(f"[GAGAL] Kamera {nama} ID {cam_id} gagal dibuka dengan {backend_name}.")
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        samakan_setting_kamera(cap, nama)

        time.sleep(1)

        berhasil = False

        for _ in range(20):
            ret, frame = cap.read()
            if ret and frame is not None:
                berhasil = True
                break
            time.sleep(0.05)

        if berhasil:
            print(f"[OK] Kamera {nama} aktif. ID {cam_id} | Backend {backend_name}")
            return cap

        print(f"[GAGAL] Kamera {nama} terbuka, tetapi frame tidak terbaca.")
        cap.release()
        time.sleep(0.3)

    return None


def open_camera_dengan_cadangan(cam_id_awal, nama, id_terlarang=None):
    cap = open_camera(cam_id_awal, nama)

    if cap is not None:
        return cap, cam_id_awal

    print(f"[INFO] Kamera {nama} ID {cam_id_awal} gagal. Mencoba ID cadangan...")

    for cam_id in CAMERA_ID_CANDIDATES:
        if cam_id == cam_id_awal:
            continue

        if id_terlarang is not None and cam_id == id_terlarang:
            continue

        cap = open_camera(cam_id, nama)

        if cap is not None:
            return cap, cam_id

    return None, None


# =========================================================
# ROI / CROP FRAME
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


# =========================================================
# PARSE HASIL DETEKSI CHARUCO
# =========================================================
def parse_hasil_detect_board(hasil):
    charuco_corners = None
    charuco_ids = None
    marker_corners = None
    marker_ids = None

    if hasil is None:
        return charuco_corners, charuco_ids, marker_corners, marker_ids

    if not isinstance(hasil, tuple):
        return charuco_corners, charuco_ids, marker_corners, marker_ids

    # OpenCV modern biasanya mengembalikan:
    # charucoCorners, charucoIds, markerCorners, markerIds
    if len(hasil) == 4:
        charuco_corners, charuco_ids, marker_corners, marker_ids = hasil

    # Cadangan jika OpenCV mengembalikan 3 nilai
    elif len(hasil) == 3:
        a, b, c = hasil

        if isinstance(a, (bool, int, float, np.bool_)):
            charuco_corners = b
            charuco_ids = c
        else:
            charuco_corners = a
            charuco_ids = b
            marker_corners = c

    # Cadangan jika OpenCV mengembalikan 2 nilai
    elif len(hasil) == 2:
        charuco_corners, charuco_ids = hasil

    return charuco_corners, charuco_ids, marker_corners, marker_ids


# =========================================================
# DETEKSI CHARUCO DEFAULT DAN LEGACY
# =========================================================
def detect_dengan_detector(detector, gray, nama_pola):
    hasil = detector.detectBoard(gray)

    charuco_corners, charuco_ids, marker_corners, marker_ids = parse_hasil_detect_board(hasil)

    jumlah_charuco = 0 if charuco_ids is None else len(charuco_ids)
    jumlah_marker = 0 if marker_ids is None else len(marker_ids)

    return {
        "pola": nama_pola,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "jumlah_charuco": jumlah_charuco,
        "jumlah_marker": jumlah_marker
    }


def detect_charuco(gray):
    hasil_default = detect_dengan_detector(
        charuco_detector_default,
        gray,
        "DEFAULT"
    )

    hasil_legacy = detect_dengan_detector(
        charuco_detector_legacy,
        gray,
        "LEGACY"
    )

    # Prioritas utama: jumlah ChArUco corner
    # Prioritas kedua: jumlah marker
    skor_default = (hasil_default["jumlah_charuco"] * 1000) + hasil_default["jumlah_marker"]
    skor_legacy = (hasil_legacy["jumlah_charuco"] * 1000) + hasil_legacy["jumlah_marker"]

    if skor_legacy > skor_default:
        return hasil_legacy
    else:
        return hasil_default


def detect_charuco_multi(gray, nama_kamera):
    if not GUNAKAN_MULTI_DETECT:
        hasil = detect_charuco(preprocess_gray(gray))
        hasil["variasi"] = "NORMAL"
        return hasil

    hasil_terbaik = None
    nama_terbaik = None
    skor_terbaik = -1

    for nama_variasi, gray_variasi in buat_variasi_gray(gray, nama_kamera):
        hasil = detect_charuco(gray_variasi)
        skor = skor_hasil_deteksi(hasil)

        if skor > skor_terbaik:
            hasil_terbaik = hasil
            nama_terbaik = nama_variasi
            skor_terbaik = skor

    hasil_terbaik["variasi"] = nama_terbaik
    return hasil_terbaik


# =========================================================
# UKURAN BOARD DI FRAME
# =========================================================
def ambil_semua_titik_deteksi(hasil):
    daftar_titik = []

    charuco_corners = hasil.get("charuco_corners")
    marker_corners = hasil.get("marker_corners")

    if charuco_corners is not None:
        daftar_titik.append(charuco_corners.reshape(-1, 2))

    if marker_corners is not None:
        try:
            marker_points = np.concatenate([m.reshape(-1, 2) for m in marker_corners], axis=0)
            daftar_titik.append(marker_points)
        except Exception:
            pass

    if not daftar_titik:
        return None

    return np.concatenate(daftar_titik, axis=0)


def hitung_rasio_board(hasil, frame_shape):
    titik = ambil_semua_titik_deteksi(hasil)

    if titik is None or len(titik) < 4:
        return 0.0, None

    h, w = frame_shape[:2]

    x_min = int(np.floor(np.min(titik[:, 0])))
    y_min = int(np.floor(np.min(titik[:, 1])))
    x_max = int(np.ceil(np.max(titik[:, 0])))
    y_max = int(np.ceil(np.max(titik[:, 1])))

    x_min = max(0, min(w - 1, x_min))
    y_min = max(0, min(h - 1, y_min))
    x_max = max(0, min(w - 1, x_max))
    y_max = max(0, min(h - 1, y_max))

    box_w = max(1, x_max - x_min)
    box_h = max(1, y_max - y_min)

    # Rasio sisi terpanjang board terhadap frame.
    rasio = max(box_w / float(w), box_h / float(h))

    return rasio, (x_min, y_min, x_max, y_max)


def status_ukuran_board(rasio):
    if rasio <= 0:
        return "UKURAN: -"
    if rasio < TARGET_BOARD_MIN:
        return f"UKURAN: {rasio*100:.0f}% | TERLALU KECIL"
    if rasio > TARGET_BOARD_MAX:
        return f"UKURAN: {rasio*100:.0f}% | TERLALU BESAR"
    return f"UKURAN: {rasio*100:.0f}% | IDEAL"


# =========================================================
# VISUALISASI
# =========================================================
def draw_target_guide(frame):
    if not TAMPILKAN_KOTAK_PANDUAN:
        return

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Kotak kecil = batas minimal 30%, kotak besar = batas maksimal 60%
    for rasio, warna, label in [
        (TARGET_BOARD_MIN, (255, 255, 0), "MIN 30%"),
        (TARGET_BOARD_MAX, (0, 255, 255), "MAX 60%")
    ]:
        bw = int(w * rasio)
        bh = int(h * rasio)
        x1 = cx - bw // 2
        y1 = cy - bh // 2
        x2 = cx + bw // 2
        y2 = cy + bh // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 1)
        cv2.putText(
            frame,
            label,
            (x1 + 5, max(95, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            warna,
            1
        )


def draw_board_bbox(frame, bbox, rasio):
    if bbox is None:
        return

    x1, y1, x2, y2 = bbox

    if TARGET_BOARD_MIN <= rasio <= TARGET_BOARD_MAX:
        warna = (0, 255, 0)
    elif rasio < TARGET_BOARD_MIN:
        warna = (0, 165, 255)
    else:
        warna = (0, 0, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)


def draw_info(frame, nama_kamera, hasil, rasio_board=0.0, crop_aktif=True):
    jumlah_charuco = hasil["jumlah_charuco"]
    jumlah_marker = hasil["jumlah_marker"]
    pola = hasil["pola"]

    if jumlah_charuco >= MIN_CHARUCO_CORNERS:
        warna = (0, 255, 0)
        status = "TERDETEKSI"
    else:
        warna = (0, 0, 255)
        status = "TIDAK TERDETEKSI"

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 88), (0, 0, 0), -1)

    teks1 = (
        f"{nama_kamera} | ChArUco: {status} | "
        f"Corners: {jumlah_charuco} | Markers: {jumlah_marker} | {pola}"
    )

    teks2 = status_ukuran_board(rasio_board)
    teks2 += " | CROP ON" if crop_aktif else " | CROP OFF"
    if "variasi" in hasil:
        teks2 += f" | DET:{hasil['variasi']}"

    cv2.putText(
        frame,
        teks1,
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        warna,
        2
    )

    cv2.putText(
        frame,
        teks2,
        (10, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        warna,
        2
    )


def draw_charuco_corners(frame, charuco_corners):
    if charuco_corners is None:
        return

    titik = charuco_corners.reshape(-1, 2)

    for p in titik:
        x, y = int(p[0]), int(p[1])
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def tampilkan_hasil(frame, nama_kamera, hasil, crop_aktif=True):
    vis = frame.copy()

    marker_corners = hasil["marker_corners"]
    marker_ids = hasil["marker_ids"]
    charuco_corners = hasil["charuco_corners"]

    if marker_corners is not None and marker_ids is not None and len(marker_ids) > 0:
        aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

    draw_charuco_corners(vis, charuco_corners)

    rasio_board, bbox = hitung_rasio_board(hasil, vis.shape)
    draw_target_guide(vis)
    draw_board_bbox(vis, bbox, rasio_board)
    draw_info(vis, nama_kamera, hasil, rasio_board, crop_aktif)

    return vis


# =========================================================
# PROGRAM UTAMA
# =========================================================
def main():
    print("\n" + "=" * 75)
    print("CAPTURE CHARUCO STEREO - LEFT BOOST MULTI DETECT")
    print("=" * 75)
    print(f"Board          : {SQUARES_X} x {SQUARES_Y}")
    print(f"Square length  : {SQUARE_LENGTH} meter / 39 mm")
    print(f"Marker length  : {MARKER_LENGTH} meter / 29 mm")
    print(f"Resolusi       : {WIDTH} x {HEIGHT}")
    print(f"FPS            : {FPS}")
    print(f"Crop aktif     : {CROP_AKTIF}")
    print(f"Simpan crop    : {SIMPAN_FRAME_CROP}")
    print(f"ROI kiri       : {ROI_LEFT}")
    print(f"ROI kanan      : {ROI_RIGHT}")
    print(f"Setup kamera   : tinggi {CAMERA_HEIGHT_CM} cm | tilt {CAMERA_TILT_MIN_DEG}-{CAMERA_TILT_MAX_DEG} derajat | FOV {CAMERA_FOV_DEG} derajat")
    print(f"Ukuran gerbong : {GERBONG_PANJANG_CM} cm x {GERBONG_LEBAR_CM} cm x {GERBONG_TINGGI_CM} cm")
    print(f"CLAHE aktif    : {GUNAKAN_CLAHE}")
    print(f"Setting kamera : {SAMAKAN_SETTING_KAMERA}")
    print(f"Sharpen deteksi: {GUNAKAN_SHARPEN_RINGAN}")
    print(f"Multi detect   : {GUNAKAN_MULTI_DETECT}")
    print(f"Left boost     : {LEFT_EXTRA_BOOST}")
    print(f"Batas sharp L/R: {MIN_SHARPNESS_LEFT} / {MIN_SHARPNESS_RIGHT}")
    print(f"Target board   : {int(TARGET_BOARD_MIN*100)}% - {int(TARGET_BOARD_MAX*100)}% sisi frame")
    print(f"Folder kiri    : {FOLDER_LEFT}")
    print(f"Folder kanan   : {FOLDER_RIGHT}")
    print("=" * 75)

    print("\n[INFO] Membuka kamera kiri...")
    capL, id_left_aktif = open_camera_dengan_cadangan(CAMERA_LEFT, "KIRI")

    if capL is None:
        print("[ERROR] Kamera kiri tidak bisa dibuka.")
        return

    print("\n[INFO] Membuka kamera kanan...")
    capR, id_right_aktif = open_camera_dengan_cadangan(
        CAMERA_RIGHT,
        "KANAN",
        id_terlarang=id_left_aktif
    )

    if capR is None:
        capL.release()
        print("[ERROR] Kamera kanan tidak bisa dibuka.")
        return

    print("\n[OK] Kamera aktif:")
    print(f"KIRI  : ID {id_left_aktif}")
    print(f"KANAN : ID {id_right_aktif}")

    print("\nTEKAN S UNTUK SIMPAN | Z UNTUK CROP ON/OFF | ESC UNTUK KELUAR\n")

    counter = 0
    crop_aktif = CROP_AKTIF

    while True:
        grabL = capL.grab()
        grabR = capR.grab()

        if not grabL or not grabR:
            print("[ERROR] Gagal grab frame dari kamera.")
            break

        retL, frameL_asli = capL.retrieve()
        retR, frameR_asli = capR.retrieve()

        if not retL or frameL_asli is None:
            print("[ERROR] Kamera kiri gagal frame.")
            break

        if not retR or frameR_asli is None:
            print("[ERROR] Kamera kanan gagal frame.")
            break

        if crop_aktif:
            frameL = crop_resize_frame(frameL_asli, ROI_LEFT)
            frameR = crop_resize_frame(frameR_asli, ROI_RIGHT)
        else:
            frameL = frameL_asli.copy()
            frameR = frameR_asli.copy()

        grayL_raw = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR_raw = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        sharpL = hitung_sharpness(grayL_raw)
        sharpR = hitung_sharpness(grayR_raw)

        # Deteksi memakai multi-preprocessing agar kamera kiri punya peluang deteksi lebih baik.
        hasil_L = detect_charuco_multi(grayL_raw, "KIRI")
        hasil_R = detect_charuco_multi(grayR_raw, "KANAN")

        visL = tampilkan_hasil(mirror_untuk_tampilan(frameL), "KAMERA KIRI", hasil_L, crop_aktif)
        visR = tampilkan_hasil(mirror_untuk_tampilan(frameR), "KAMERA KANAN", hasil_R, crop_aktif)

        status_sharp_L = "TAJAM" if sharpL >= MIN_SHARPNESS_LEFT else "BLUR"
        status_sharp_R = "TAJAM" if sharpR >= MIN_SHARPNESS_RIGHT else "BLUR"
        warna_sharp_L = (0, 255, 0) if sharpL >= MIN_SHARPNESS_LEFT else (0, 0, 255)
        warna_sharp_R = (0, 255, 0) if sharpR >= MIN_SHARPNESS_RIGHT else (0, 0, 255)

        cv2.putText(visL, f"Sharp: {sharpL:.1f} | {status_sharp_L}", (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.48, warna_sharp_L, 2)
        cv2.putText(visR, f"Sharp: {sharpR:.1f} | {status_sharp_R}", (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.48, warna_sharp_R, 2)

        combined = np.hstack((visL, visR))

        cv2.imshow("CAPTURE CHARUCO STEREO LEFT BOOST MULTI DETECT", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("z") or key == ord("Z"):
            crop_aktif = not crop_aktif
            print(f"[INFO] Crop sekarang: {'ON' if crop_aktif else 'OFF'}")

        if key == ord("s") or key == ord("S"):
            jumlah_L = hasil_L["jumlah_charuco"]
            jumlah_R = hasil_R["jumlah_charuco"]

            rasio_L, _ = hitung_rasio_board(hasil_L, frameL.shape)
            rasio_R, _ = hitung_rasio_board(hasil_R, frameR.shape)

            ukuran_L_ok = TARGET_BOARD_MIN <= rasio_L <= TARGET_BOARD_MAX
            ukuran_R_ok = TARGET_BOARD_MIN <= rasio_R <= TARGET_BOARD_MAX
            sharp_L_ok = sharpL >= MIN_SHARPNESS_LEFT
            sharp_R_ok = sharpR >= MIN_SHARPNESS_RIGHT

            if BLOKIR_SIMPAN_JIKA_BLUR and (not sharp_L_ok or not sharp_R_ok):
                print("[GAGAL] Frame belum disimpan karena salah satu kamera masih blur.")
                print(f"        Sharp kiri  : {sharpL:.1f} | {'OK' if sharp_L_ok else 'BLUR'}")
                print(f"        Sharp kanan : {sharpR:.1f} | {'OK' if sharp_R_ok else 'BLUR'}")
                continue

            if jumlah_L >= MIN_CHARUCO_CORNERS and jumlah_R >= MIN_CHARUCO_CORNERS:
                counter += 1

                path_left = os.path.join(FOLDER_LEFT, f"frame_{counter:03d}.png")
                path_right = os.path.join(FOLDER_RIGHT, f"frame_{counter:03d}.png")

                if crop_aktif and SIMPAN_FRAME_CROP:
                    simpanL = frameL
                    simpanR = frameR
                    mode_simpan = "CROP/ROI"
                else:
                    simpanL = frameL_asli
                    simpanR = frameR_asli
                    mode_simpan = "FULL FRAME"

                cv2.imwrite(path_left, simpanL)
                cv2.imwrite(path_right, simpanR)

                print(f"[OK] Frame {counter:03d} tersimpan. Mode: {mode_simpan}")
                print(f"     KIRI  : {path_left}")
                print(f"     KANAN : {path_right}")
                print(f"     Corners kiri  : {jumlah_L}")
                print(f"     Corners kanan : {jumlah_R}")
                print(f"     Ukuran kiri   : {rasio_L*100:.1f}% | {'IDEAL' if ukuran_L_ok else 'CEK POSISI'}")
                print(f"     Ukuran kanan  : {rasio_R*100:.1f}% | {'IDEAL' if ukuran_R_ok else 'CEK POSISI'}")
                print(f"     Sharp kiri    : {sharpL:.1f} | {'OK' if sharp_L_ok else 'BLUR'}")
                print(f"     Sharp kanan   : {sharpR:.1f} | {'OK' if sharp_R_ok else 'BLUR'}")
                print(f"     Pola kiri     : {hasil_L['pola']} | Deteksi: {hasil_L.get('variasi', '-')}")
                print(f"     Pola kanan    : {hasil_R['pola']} | Deteksi: {hasil_R.get('variasi', '-')}")

                if not (ukuran_L_ok and ukuran_R_ok):
                    print("     [CATATAN] Board terdeteksi, tetapi ukurannya belum 30-60% di salah satu kamera.")
                    print("               Data tetap disimpan. Untuk dataset final, lebih baik ulangi dengan ukuran IDEAL.")

            else:
                print("[GAGAL] ChArUco belum cukup terdeteksi di kedua kamera.")
                print(f"        Kiri  - Corners: {jumlah_L}, Markers: {hasil_L['jumlah_marker']}, Ukuran: {rasio_L*100:.1f}%, Pola: {hasil_L['pola']}")
                print(f"        Kanan - Corners: {jumlah_R}, Markers: {hasil_R['jumlah_marker']}, Ukuran: {rasio_R*100:.1f}%, Pola: {hasil_R['pola']}")

        if key == 27:
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

    print("\nSELESAI")
    print(f"TOTAL DATASET: {counter}")


if __name__ == "__main__":
    main()
