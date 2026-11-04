"""
SCRIPT 5 — DISPARITY MAP → DEPTH MAP → Z_REF (MODE UKUR)
==============================================
Skripsi: Estimasi Massa Muatan Batubara (Enina Devianti Farid - UNAIR)

Fungsi  : 
  1. Buka kamera stereo (sudah terkalibrasi)
  2. Remap frame menggunakan stereomap.xml
  3. Hitung disparity map dengan StereoSGBM
  4. Konversi ke depth map 3D menggunakan matriks Q (cv2.reprojectImageTo3D)
  5. Filter noise dengan Median + WLS Filter
  6. Mode Z_ref: ukur jarak lantai gerbong kosong → simpan sebagai referensi

Cara pakai:
    pip install opencv-contrib-python numpy
    python 05_depth_map.py

Kontrol keyboard:
    [D]   → Toggle tampilan: depth map / frame asli / disparity
    [Z]   → Masuk mode Z_REF (arahkan ke gerbong KOSONG, tekan lagi untuk simpan)
    [S]   → Simpan snapshot depth map saat ini
    [ESC] → Keluar

Output:
    hasil_kalibrasi/zref.npz   ← nilai Z referensi lantai gerbong kosong
"""

import cv2
import numpy as np
import os
import warnings
from collections import deque

# ══════════════════════════════════════════
# KONFIGURASI — SESUAIKAN DI SINI
# ══════════════════════════════════════════

CAMERA_ID_KIRI  = 0
CAMERA_ID_KANAN = 1

# =========================================================
# SETUP SELARAS SCRIPT 06 / SCRIPT 4
# =========================================================
# Karena kalibrasi kamu memakai crop dari Script 06, frame live di Script 5
# juga wajib di-crop dengan ROI yang sama SEBELUM cv2.remap.
TERAPKAN_CROP_REALTIME = True
ROI_LEFT = (0.00, 0.05, 1.00, 0.95)
ROI_RIGHT = (0.00, 0.05, 1.00, 0.95)

CAMERA_RAW_WIDTH = 640
CAMERA_RAW_HEIGHT = 480
FPS = 30

CAMERA_HEIGHT_CM = 60.0
CAMERA_TILT_MIN_DEG = 10.0
CAMERA_TILT_MAX_DEG = 15.0
CAMERA_FOV_DEG = 120.0

# Koreksi sederhana untuk membandingkan depth sumbu kamera dengan tinggi vertikal manual.
# Jika kamera tilt 10-15 derajat, jarak sepanjang sumbu kamera memang lebih besar
# daripada pengukuran vertikal. Untuk crosscheck manual, gunakan koreksi cos(tilt).
KOREKSI_TILT_AKTIF = True
CAMERA_TILT_ESTIMASI_DEG = 12.5

GERBONG_PANJANG_CM = 40.0
GERBONG_LEBAR_CM = 10.0
GERBONG_TINGGI_CM = 7.5

RMS_STEREO_BATAS_AMAN = 1.0

# Ukuran dashboard tampilan OpenCV. Dibuat lebih lebar agar teks tidak terpotong.
# Jika layar laptop kecil, OpenCV tetap bisa di-resize manual.
DISPLAY_CANVAS_WIDTH = 1100
DISPLAY_CANVAS_HEIGHT = 720
DISPLAY_HEADER_HEIGHT = 172
DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 740
DISPLAY_MIRROR_HORIZONTAL = True

# Path dibuat relatif ke LOKASI FILE SCRIPT ini, bukan ke folder tempat
# command dijalankan -- supaya tidak perlu cd ke folder tertentu dulu
# sebelum menjalankan script (konsisten dengan Script 11).
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
STEREOMAP_XML   = os.path.join(BASE_DIR, "hasil_kalibrasi", "stereomap.xml")
OUTPUT_DIR      = os.path.join(BASE_DIR, "hasil_kalibrasi")

# ── Parameter StereoSGBM ──
# Sesuaikan setelah melihat hasil visual — tidak ada nilai "pasti"
# Panduan awal untuk webcam 1280×720:
SGBM_MIN_DISP    = 0        # biasanya 0
SGBM_NUM_DISP    = 96       # harus kelipatan 16; naikkan jika objek dekat
SGBM_BLOCK_SIZE  = 5        # ganjil, 3–11; lebih kecil = lebih detail tapi lebih noisy
SGBM_P1          = 8  * 3 * SGBM_BLOCK_SIZE ** 2   # penalti kehalusan 1
SGBM_P2          = 32 * 3 * SGBM_BLOCK_SIZE ** 2   # penalti kehalusan 2 (harus > P1)
SGBM_DISP12_DIFF = 1        # toleransi konsistensi kiri-kanan
SGBM_UNIQUENESS  = 10       # filter piksel ambigu (%)
SGBM_SPECKLE_WS  = 100      # ukuran window filter speckle
SGBM_SPECKLE_RNG = 32       # range filter speckle

# ── Parameter WLS Filter ──
WLS_LAMBDA   = 8000.0   # kehalusan (makin besar makin halus, tapi bisa blur tepi)
WLS_SIGMA    = 1.5      # sensitivitas tepi

# ── Parameter Z_ref ──
# Area tengah frame yang dipakai untuk ukur lantai gerbong (dalam persen)
ZREF_CENTER_CROP = 0.5  # pakai area 50% tengah frame

# ── Batas jarak valid (meter) ──
# Bug tanda negatif pada depth_map SUDAH DIPERBAIKI (lihat disp_ke_depth).
# Sekarang rentang ini bisa dikembalikan ke nilai yang sesuai jarak fisik
# aktual: kamera ke lantai gerbong sekitar 50-60cm.
DEPTH_MIN_M = 0.20   # 20 cm -- margin aman
DEPTH_MAX_M = 0.90   # 90 cm -- margin aman di atas 60cm

# ── Mode diagnostik tambahan ──
DIAGNOSTIK_AKTIF = True  # set False setelah masalah selesai dianalisis


# ── ROBUST MODE / STABILISASI SCRIPT 05 ──
# Tujuan: depth/Z_ref tidak diambil dari 1 frame saja, tetapi dari median beberapa frame.
# Ini membuat hasil lebih stabil saat ada noise kecil, bayangan, atau matching SGBM sesaat yang buruk.
STABILISASI_AKTIF = True
STABILISASI_N_FRAME = 15       # jumlah frame untuk median temporal; 10-20 umum dipakai
STABILISASI_MIN_FRAME = 8      # minimal frame sebelum status boleh dianggap stabil

# Batas kelayakan area tengah/ROI untuk Z_ref dan snapshot.
# ROI yang dimaksud di sini adalah area tengah ZREF_CENTER_CROP, bukan ROI evaluasi 05B manual.
STABIL_MIN_VALID_ROI = 85.0    # % piksel valid minimal di ROI tengah
STABIL_MAX_ROI_STD_CM = 1.50   # INFO saja: std spasial ROI. Untuk objek/balok nilai ini memang bisa besar.
STABIL_MAX_JITTER_CM = 0.80    # perubahan median ROI antar-frame maksimal

# FIX STABILITAS:
# Script versi sebelumnya menganggap TIDAK STABIL jika stdROI spasial > 1.50 cm.
# Itu cocok untuk lantai kosong yang rata, tetapi SALAH/terlalu ketat saat ada balok/objek,
# karena area ROI memang berisi beberapa ketinggian berbeda sehingga stdROI besar.
# Jadi status stabil utama sekarang memakai validROI + jitter temporal, bukan stdROI spasial.
STABIL_GUNAKAN_STD_SPASIAL_SEBAGAI_SYARAT = False

# Simpan tetap boleh dipaksa, tetapi script akan memberi WARNING kalau belum stabil.
WAJIB_STABIL_UNTUK_SIMPAN = False

# Mode penandaan snapshot. Ini hanya metadata dan panduan tampilan;
# pemilihan ROI objek tetap dilakukan lebih teliti di Script 05B.
SNAPSHOT_MODES = [
    ("LANTAI_KOSONG", "LANTAI KOSONG", "Untuk Z_ref / evaluasi jarak lantai. Gerbong harus kosong."),
    ("OBJEK_BALOK", "OBJEK BALOK", "Untuk evaluasi tinggi balok. ROI balok dipilih di Script 05B."),
    ("MUATAN_BATUBARA", "MUATAN BATUBARA", "Untuk tahap lanjut integrasi U-Net, volume, dan massa."),
]

# Penguncian kamera. Tidak semua webcam mendukung semua properti ini.
# Jika tidak didukung, OpenCV biasanya mengabaikan tanpa error fatal.
KUNCI_SETTING_KAMERA = True
MATIKAN_AUTO_WB = True
MATIKAN_AUTOFOCUS = True
WB_TEMPERATURE = 4500
BRIGHTNESS_VALUE = 128
CONTRAST_VALUE = 128

# Exposure manual dibuat opsional karena nilai exposure tiap webcam berbeda-beda.
# Aktifkan hanya kalau gambar terlalu berubah-ubah dan kamera mendukung properti exposure.
PAKAI_EXPOSURE_MANUAL = False
EXPOSURE_MANUAL_VALUE = -6
GAIN_MANUAL_VALUE = 0

# ══════════════════════════════════════════


def _node_empty(node):
    try:
        return node.empty()
    except Exception:
        return True


def baca_float_xml(fs, nama_node, default=None):
    node = fs.getNode(nama_node)
    if _node_empty(node):
        return default
    try:
        return float(node.real())
    except Exception:
        return default


def baca_int_xml(fs, nama_node, default=None):
    val = baca_float_xml(fs, nama_node, default=None)
    if val is None:
        return default
    return int(round(val))


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
    if frame is None:
        return None
    if TERAPKAN_CROP_REALTIME:
        return crop_resize_frame(frame, pilih_roi(nama_kamera), image_size)
    return frame


def pastikan_ukuran_frame(frame, image_size, nama_kamera):
    target_w, target_h = image_size
    h, w = frame.shape[:2]
    if (w, h) == image_size:
        return frame
    print(f"[WARNING] Ukuran frame {nama_kamera} = {(w, h)}, di-resize ke ukuran kalibrasi {image_size}.")
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def resize_agar_muatt_layar(img, max_width=DISPLAY_MAX_WIDTH, max_height=DISPLAY_MAX_HEIGHT):
    h, w = img.shape[:2]
    scale = min(max_width / float(w), max_height / float(h), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return img


def resize_fit_area(img, max_width, max_height):
    """Resize gambar agar muat di area dashboard tanpa terpotong."""
    h, w = img.shape[:2]
    scale = min(max_width / float(w), max_height / float(h))
    scale = max(0.05, scale)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if (new_w, new_h) != (w, h):
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    else:
        resized = img.copy()
    return resized, scale


def mirror_untuk_tampilan(img):
    return cv2.flip(img, 1) if DISPLAY_MIRROR_HORIZONTAL else img


def put_text_outline(img, text, org, font_scale=0.55, color=(255, 255, 255), thickness=1):
    """Tulis teks dengan outline hitam agar tetap terbaca di background terang/gelap."""
    x, y = org
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def buat_dashboard_display(tampilan, label_mode_text, sumber_depth, persen_valid, persen_valid_roi,
                           stabil_ok, stabil_reason, zref_aktif, mode_zref=False,
                           hasil_zref=None, stats_roi=None, snapshot_label="LANTAI KOSONG",
                           snapshot_code="LANTAI_KOSONG"):
    """
    Buat tampilan dashboard terpisah: panel teks di atas + gambar depth di bawah.
    Tujuan: informasi tidak lagi ditulis di atas frame 640px yang sempit, sehingga tidak terpotong.
    """
    canvas_w = int(DISPLAY_CANVAS_WIDTH)
    canvas_h = int(DISPLAY_CANVAS_HEIGHT)
    header_h = int(DISPLAY_HEADER_HEIGHT)
    pad = 8

    # Background abu gelap agar area kosong tidak putih/menyilaukan.
    dashboard = np.full((canvas_h, canvas_w, 3), (45, 45, 45), dtype=np.uint8)
    dashboard[:header_h, :] = (0, 0, 0)

    # Resize image agar seluruh depth/frame terlihat utuh.
    area_w = canvas_w - 2 * pad
    area_h = canvas_h - header_h - 2 * pad
    img_resized, scale = resize_fit_area(mirror_untuk_tampilan(tampilan), area_w, area_h)
    ih, iw = img_resized.shape[:2]
    xoff = (canvas_w - iw) // 2
    yoff = header_h + max(0, (area_h - ih) // 2)
    dashboard[yoff:yoff+ih, xoff:xoff+iw] = img_resized

    warna_valid = (0, 255, 0) if persen_valid >= 90 else (0, 165, 255) if persen_valid >= 70 else (0, 0, 255)
    warna_stabil = (0, 255, 0) if stabil_ok else (0, 180, 255)

    # Baris 1-3: teks dibuat pendek agar tidak terpotong.
    put_text_outline(
        dashboard,
        f"Mode: {label_mode_text} | {sumber_depth}",
        (10, 28), 0.66, warna_valid, 2
    )
    put_text_outline(
        dashboard,
        f"Valid total: {persen_valid:.1f}% | ROI tengah: {persen_valid_roi:.1f}%",
        (10, 58), 0.58, warna_valid, 1
    )
    reason_short = str(stabil_reason)[:105]
    put_text_outline(
        dashboard,
        f"Stabil: {'YA' if stabil_ok else 'BELUM'} | {reason_short}",
        (10, 86), 0.53, warna_stabil, 1
    )

    if zref_aktif is not None:
        zref_vert = koreksi_tilt_z(zref_aktif)
        ztxt = f"Z_ref TERSIMPAN raw={zref_aktif*100:.2f} cm | vert={zref_vert*100:.2f} cm | [Z]=ukur/update  [S]=snapshot  [F]=reset"
        zcolor = (210, 210, 210)
    else:
        ztxt = "Z_ref: BELUM DIUKUR | kosongkan gerbong lalu tekan [Z]"
        zcolor = (0, 180, 255)
    put_text_outline(dashboard, ztxt[:126], (10, 111), 0.43, zcolor, 1)

    if mode_zref:
        operasi = "OPERASI: UKUR Z_REF | GERBONG WAJIB KOSONG | tekan [Z] lagi untuk simpan"
        warna_operasi = (0, 255, 255)
    else:
        operasi = f"OPERASI: SNAPSHOT {snapshot_label} | [M]=ganti jenis | [S]=simpan"
        warna_operasi = (0, 255, 0) if snapshot_code == "LANTAI_KOSONG" else (0, 220, 255) if snapshot_code == "OBJEK_BALOK" else (255, 220, 0)
    put_text_outline(dashboard, operasi[:126], (10, 136), 0.43, warna_operasi, 1)

    # Tampilkan median ROI tengah saat ini supaya tidak bingung:
    # angka ini adalah median area tengah besar, BUKAN otomatis median permukaan balok.
    roi_text = "ROI tengah sekarang: belum cukup data"
    try:
        med_cm = float(stats_roi.get("median_cm", np.nan)) if stats_roi is not None else np.nan
        if np.isfinite(med_cm):
            roi_text = f"ROI tengah median={med_cm:.2f} cm"
            if zref_aktif is not None and snapshot_code == "OBJEK_BALOK":
                h_tmp = zref_aktif * 100.0 - med_cm
                roi_text += f" | tinggi sementara ROI={h_tmp:.2f} cm (valid hanya jika ROI tepat di balok)"
            elif zref_aktif is not None and snapshot_code == "LANTAI_KOSONG":
                e_tmp = med_cm - zref_aktif * 100.0
                roi_text += f" | beda dari Z_ref={e_tmp:+.2f} cm"
    except Exception:
        pass
    put_text_outline(dashboard, roi_text[:126], (10, 160), 0.40, (230, 230, 230), 1)

    # ROI Z_ref digambar pada area gambar yang sudah di-resize, bukan menimpa header.
    if mode_zref and hasil_zref is not None:
        try:
            z_now, r1, r2, c1, c2 = hasil_zref
            x1 = xoff + int(round(c1 * scale))
            y1 = yoff + int(round(r1 * scale))
            x2 = xoff + int(round(c2 * scale))
            y2 = yoff + int(round(r2 * scale))
            cv2.rectangle(dashboard, (x1, y1), (x2, y2), (0, 255, 255), 2)
            z_vert = koreksi_tilt_z(z_now)
            label = f"Z raw {z_now*100:.2f} cm | vert {z_vert*100:.2f} cm | tekan [Z] simpan"
            label_y = max(header_h + 20, y1 - 10)
            put_text_outline(dashboard, label, (x1, label_y), 0.52, (0, 255, 255), 1)
        except Exception:
            pass

    return resize_agar_muatt_layar(dashboard)


def set_prop_aman(cap, prop, value, nama_prop=""):
    """Set property kamera dengan aman. Banyak webcam mengabaikan sebagian property."""
    try:
        ok = cap.set(prop, value)
        return bool(ok)
    except Exception:
        return False


def kunci_setting_kamera(cap, nama="kamera"):
    """Kurangi perubahan frame akibat auto exposure/white balance/autofocus."""
    if cap is None or not KUNCI_SETTING_KAMERA:
        return

    set_prop_aman(cap, cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS_VALUE, "BRIGHTNESS")
    set_prop_aman(cap, cv2.CAP_PROP_CONTRAST, CONTRAST_VALUE, "CONTRAST")
    set_prop_aman(cap, cv2.CAP_PROP_GAIN, float(GAIN_MANUAL_VALUE), "GAIN")

    # Nilai auto exposure berbeda antar backend/driver.
    # 0.25 sering berarti manual pada Windows/DirectShow, 0.75 sering auto.
    # Agar tidak membuat gambar mendadak gelap, auto exposure hanya dimatikan
    # jika PAKAI_EXPOSURE_MANUAL=True.
    if PAKAI_EXPOSURE_MANUAL:
        set_prop_aman(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "AUTO_EXPOSURE")
        set_prop_aman(cap, cv2.CAP_PROP_EXPOSURE, float(EXPOSURE_MANUAL_VALUE), "EXPOSURE")
        set_prop_aman(cap, cv2.CAP_PROP_GAIN, float(GAIN_MANUAL_VALUE), "GAIN")

    if MATIKAN_AUTO_WB:
        set_prop_aman(cap, cv2.CAP_PROP_AUTO_WB, 0, "AUTO_WB")
        if hasattr(cv2, "CAP_PROP_WB_TEMPERATURE"):
            set_prop_aman(cap, cv2.CAP_PROP_WB_TEMPERATURE, WB_TEMPERATURE, "WB_TEMPERATURE")

    if MATIKAN_AUTOFOCUS:
        set_prop_aman(cap, cv2.CAP_PROP_AUTOFOCUS, 0, "AUTOFOCUS")


def buka_kamera(camera_id, nama="kamera"):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(CAMERA_RAW_WIDTH))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(CAMERA_RAW_HEIGHT))
    cap.set(cv2.CAP_PROP_FPS, int(FPS))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    kunci_setting_kamera(cap, nama)
    return cap


def info_q(Q):
    if Q is None:
        return None, None
    try:
        focal = float(abs(Q[2, 3]))
        baseline = float(abs(1.0 / Q[3, 2])) if abs(Q[3, 2]) > 1e-12 else None
        return focal, baseline
    except Exception:
        return None, None


def muat_stereomap():
    """Muat stereo maps dan matriks Q dari file XML."""
    if not os.path.exists(STEREOMAP_XML):
        print(f"[ERROR] {STEREOMAP_XML} tidak ditemukan!")
        print(f"        Jalankan dulu Script 3 dan Script 4 sampai stereomap.xml valid.")
        return None

    f = cv2.FileStorage(STEREOMAP_XML, cv2.FILE_STORAGE_READ)
    mapLx = f.getNode("stereoMapL_x").mat()
    mapLy = f.getNode("stereoMapL_y").mat()
    mapRx = f.getNode("stereoMapR_x").mat()
    mapRy = f.getNode("stereoMapR_y").mat()
    Q = f.getNode("Q").mat()

    if mapLx is None or mapLy is None or mapRx is None or mapRy is None or Q is None:
        f.release()
        print("[ERROR] stereomap.xml tidak lengkap. Pastikan Script 3/4 menghasilkan stereoMap dan Q.")
        return None

    image_width = baca_int_xml(f, "image_width", default=mapLx.shape[1])
    image_height = baca_int_xml(f, "image_height", default=mapLx.shape[0])
    rms_stereo = baca_float_xml(f, "rms_stereo", default=None)
    valid_pairs = baca_int_xml(f, "valid_pairs", default=None)
    baseline_m = baca_float_xml(f, "baseline_m", default=None)
    f.release()

    data = {
        "mapLx": mapLx,
        "mapLy": mapLy,
        "mapRx": mapRx,
        "mapRy": mapRy,
        "Q": Q,
        "image_size": (image_width, image_height),
        "rms_stereo": rms_stereo,
        "valid_pairs": valid_pairs,
        "baseline_m": baseline_m,
    }

    focal_q, baseline_q = info_q(Q)

    print(f"  [✓] Stereo maps & matriks Q dimuat dari {STEREOMAP_XML}")
    print(f"  [✓] Map size: {(image_width, image_height)}")
    if rms_stereo is not None:
        print(f"  [✓] RMS stereo: {rms_stereo:.6f} px")
        if rms_stereo > RMS_STEREO_BATAS_AMAN:
            print(f"  [PERINGATAN] RMS stereo > {RMS_STEREO_BATAS_AMAN} px. Depth map hanya untuk cek visual, belum final.")
    if valid_pairs is not None:
        print(f"  [✓] Valid pairs: {valid_pairs}")
    if baseline_m is not None:
        print(f"  [✓] Baseline XML: {baseline_m*100:.3f} cm")
    if focal_q is not None and baseline_q is not None:
        print(f"  [✓] Q info: focal≈{focal_q:.3f} px | baseline≈{baseline_q*100:.3f} cm")
    print(f"  [✓] Matriks Q:\n{Q}")
    return data


def buat_sgbm():
    """Buat objek StereoSGBM dan matcher kanan untuk WLS."""
    left_matcher = cv2.StereoSGBM_create(
        minDisparity    = SGBM_MIN_DISP,
        numDisparities  = SGBM_NUM_DISP,
        blockSize       = SGBM_BLOCK_SIZE,
        P1              = SGBM_P1,
        P2              = SGBM_P2,
        disp12MaxDiff   = SGBM_DISP12_DIFF,
        uniquenessRatio = SGBM_UNIQUENESS,
        speckleWindowSize  = SGBM_SPECKLE_WS,
        speckleRange       = SGBM_SPECKLE_RNG,
        mode            = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    # Matcher kanan (mirror dari kiri) — diperlukan WLS filter
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    return left_matcher, right_matcher


def hitung_disparity_wls(gray_kiri, gray_kanan, left_matcher, right_matcher):
    """
    Hitung disparity map dengan SGBM + WLS filter.
    Return: disparity float32 (satuan piksel, sudah dibagi 16)
    """
    disp_L = left_matcher.compute(gray_kiri,  gray_kanan)
    disp_R = right_matcher.compute(gray_kanan, gray_kiri)

    # WLS filter — menghaluskan + mengisi lubang (hole filling)
    wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(WLS_LAMBDA)
    wls.setSigmaColor(WLS_SIGMA)
    disp_filtered = wls.filter(disp_L, gray_kiri, disparity_map_right=disp_R)

    # SGBM menghasilkan nilai x16 — bagi 16 untuk dapat nilai piksel asli
    disp_float = disp_filtered.astype(np.float32) / 16.0
    return disp_float


def disp_ke_depth(disp_float, Q):
    """
    Konversi disparity map → point cloud 3D menggunakan matriks Q.
    Return: depth_map (float32, satuan METER), valid_mask

    PENTING -- BUG YANG SUDAH DIPERBAIKI:
    reprojectImageTo3D menghasilkan nilai Z NEGATIF dengan konvensi
    matriks Q tertentu (tergantung tanda elemen Q[3,2]). Sebelumnya kode
    ini TIDAK mengambil nilai absolut, sehingga depth_map selalu negatif
    (misal -0.93m), dan filter "depth_map > DEPTH_MIN_M" (positif) SELALU
    gagal karena membandingkan negatif dengan positif -- ini sebabnya
    valid_mask selalu 0.0% walau disparity aktualnya sudah benar.
    Diperbaiki dengan np.abs(), konsisten dengan Script 11 yang sudah
    memakai USE_ABS_DEPTH=True untuk masalah yang sama.
    """
    points_3d = cv2.reprojectImageTo3D(disp_float, Q)

    # Ambil channel Z saja (kedalaman), lalu absolutkan -- lihat catatan di atas
    depth_map = np.abs(points_3d[:, :, 2])

    # Mask piksel valid: disparitas > 0, jarak dalam rentang wajar
    # reprojectImageTo3D mengembalikan nilai sangat besar untuk piksel invalid
    valid_mask = (
        (disp_float > SGBM_MIN_DISP + 1) &
        np.isfinite(depth_map) &
        (depth_map > DEPTH_MIN_M) &
        (depth_map < DEPTH_MAX_M)
    )

    # Set piksel tidak valid ke NaN agar tidak mengganggu perhitungan
    depth_clean = depth_map.copy()
    depth_clean[~valid_mask] = np.nan

    return depth_clean, valid_mask


def median_filter(depth_map, kernel=5):
    """
    Terapkan median filter pada depth map.
    NaN diganti 0 sebelum filter, lalu dikembalikan.
    """
    temp = depth_map.copy()
    nan_mask = np.isnan(temp)

    if np.all(nan_mask):
        return depth_map.copy()

    max_depth = float(np.nanmax(depth_map))
    if not np.isfinite(max_depth) or max_depth <= 0:
        return depth_map.copy()

    temp[nan_mask] = 0

    # Median filter bekerja pada uint8/uint16 — normalisasi sementara
    temp_norm = cv2.normalize(temp, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
    temp_filtered = cv2.medianBlur(temp_norm, kernel)

    # Kembalikan ke skala asli
    result = temp_filtered.astype(np.float32) / 65535.0 * max_depth
    result[nan_mask] = np.nan
    return result


def hitung_persen_valid(valid_mask):
    """Hitung persentase piksel valid."""
    return 100.0 * np.sum(valid_mask) / valid_mask.size


def area_tengah_indices(shape, crop):
    h, w = shape[:2]
    r1 = int(h * (0.5 - crop/2))
    r2 = int(h * (0.5 + crop/2))
    c1 = int(w * (0.5 - crop/2))
    c2 = int(w * (0.5 + crop/2))
    return r1, r2, c1, c2


def hitung_persen_valid_area_tengah(valid_mask, crop):
    r1, r2, c1, c2 = area_tengah_indices(valid_mask.shape, crop)
    area_mask = valid_mask[r1:r2, c1:c2]
    if area_mask.size == 0:
        return 0.0
    return 100.0 * np.sum(area_mask) / area_mask.size

def statistik_roi_depth(depth_map, valid_mask, crop):
    """Statistik area tengah untuk menilai stabil/tidaknya depth."""
    r1, r2, c1, c2 = area_tengah_indices(depth_map.shape, crop)
    area_depth = depth_map[r1:r2, c1:c2]
    area_mask = valid_mask[r1:r2, c1:c2] & np.isfinite(area_depth)

    if area_mask.size == 0:
        return {
            "r1": r1, "r2": r2, "c1": c1, "c2": c2,
            "valid_roi": 0.0, "median_m": np.nan, "median_cm": np.nan,
            "std_cm": np.nan, "jumlah_valid": 0
        }

    jumlah_valid = int(np.sum(area_mask))
    valid_roi = 100.0 * jumlah_valid / area_mask.size
    if jumlah_valid == 0:
        median_m = np.nan
        std_cm = np.nan
    else:
        nilai = area_depth[area_mask]
        median_m = float(np.nanmedian(nilai))
        std_cm = float(np.nanstd(nilai) * 100.0)

    return {
        "r1": r1, "r2": r2, "c1": c1, "c2": c2,
        "valid_roi": float(valid_roi),
        "median_m": median_m,
        "median_cm": float(median_m * 100.0) if np.isfinite(median_m) else np.nan,
        "std_cm": std_cm,
        "jumlah_valid": jumlah_valid,
    }


def hitung_depth_temporal_median(depth_buffer):
    """Gabungkan beberapa depth map menjadi 1 depth map stabil dengan median temporal."""
    if (not STABILISASI_AKTIF) or len(depth_buffer) < STABILISASI_MIN_FRAME:
        return None, None

    stack = np.stack(list(depth_buffer), axis=0).astype(np.float32)
    finite_count = np.sum(np.isfinite(stack), axis=0)

    # Piksel dianggap valid kalau valid minimal pada 40% frame buffer.
    min_count = max(3, int(np.ceil(0.40 * len(depth_buffer))))
    valid_temporal = finite_count >= min_count

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        depth_temporal = np.nanmedian(stack, axis=0).astype(np.float32)

    depth_temporal[~valid_temporal] = np.nan
    return depth_temporal, valid_temporal


def status_stabil(stats_roi, riwayat_median_cm, n_frame_buffer):
    """
    Tentukan apakah depth cukup stabil untuk Z_ref/snapshot.

    Catatan penting:
    - validROI menunjukkan berapa banyak piksel depth yang valid di ROI tengah.
    - jitter menunjukkan perubahan median depth antar-frame. Ini indikator stabilitas waktu.
    - stdROI menunjukkan variasi kedalaman DI DALAM area ROI. Nilai ini bisa besar kalau ada balok/objek,
      dinding, atau permukaan tidak rata. Karena itu stdROI tidak dipakai sebagai syarat utama,
      kecuali STABIL_GUNAKAN_STD_SPASIAL_SEBAGAI_SYARAT=True.
    """
    if n_frame_buffer < STABILISASI_MIN_FRAME:
        return False, f"buffer {n_frame_buffer}/{STABILISASI_MIN_FRAME}"

    valid_roi = stats_roi.get("valid_roi", 0.0)
    std_cm = stats_roi.get("std_cm", np.nan)

    median_valid = [x for x in riwayat_median_cm if np.isfinite(x)]
    if len(median_valid) >= 2:
        jitter_cm = float(np.max(median_valid) - np.min(median_valid))
    else:
        jitter_cm = np.nan

    alasan = []
    if valid_roi < STABIL_MIN_VALID_ROI:
        alasan.append(f"validROI {valid_roi:.1f}% < {STABIL_MIN_VALID_ROI:.0f}%")

    if (not np.isfinite(jitter_cm)) or jitter_cm > STABIL_MAX_JITTER_CM:
        alasan.append(f"jitter {jitter_cm:.2f}cm > {STABIL_MAX_JITTER_CM:.2f}cm")

    # Opsional saja. Untuk lantai kosong boleh diaktifkan, tetapi untuk balok/objek jangan,
    # karena objek memang membuat stdROI besar walaupun frame sudah stabil.
    if STABIL_GUNAKAN_STD_SPASIAL_SEBAGAI_SYARAT:
        if (not np.isfinite(std_cm)) or std_cm > STABIL_MAX_ROI_STD_CM:
            alasan.append(f"stdROI {std_cm:.2f}cm > {STABIL_MAX_ROI_STD_CM:.2f}cm")

    if alasan:
        info_std = f" | stdROI(info) {std_cm:.2f}cm" if np.isfinite(std_cm) else ""
        return False, "; ".join(alasan) + info_std

    return True, f"OK | validROI {valid_roi:.1f}% | jitter {jitter_cm:.2f}cm | stdROI(info) {std_cm:.2f}cm"


def koreksi_tilt_z(z_m):
    if z_m is None or not KOREKSI_TILT_AKTIF:
        return z_m
    theta = np.deg2rad(float(CAMERA_TILT_ESTIMASI_DEG))
    return float(z_m) * float(np.cos(theta))


def depth_ke_colormap(depth_map, valid_mask):
    """Konversi depth map ke colormap BGR untuk ditampilkan."""
    disp_vis = np.zeros(depth_map.shape, dtype=np.uint8)
    if np.any(valid_mask):
        d_min = np.nanmin(depth_map[valid_mask])
        d_max = np.nanmax(depth_map[valid_mask])
        if d_max > d_min:
            normalized = (depth_map - d_min) / (d_max - d_min)
            normalized = np.clip(normalized, 0, 1)
            normalized[~valid_mask] = 0
            disp_vis = (normalized * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
    colormap[~valid_mask] = [20, 20, 20]  # area invalid = abu gelap
    return colormap


def ukur_zref(depth_map, valid_mask):
    """
    Ukur Z_ref dari area tengah frame (lantai gerbong kosong).
    Return: nilai rata-rata Z dalam meter, atau None jika tidak cukup data.
    """
    crop = ZREF_CENTER_CROP
    r1, r2, c1, c2 = area_tengah_indices(depth_map.shape, crop)

    area_depth = depth_map[r1:r2, c1:c2]
    area_mask  = valid_mask[r1:r2, c1:c2]

    persen_valid = 100.0 * np.sum(area_mask) / area_mask.size
    if persen_valid < 50:
        print(f"  [WARNING] Area tengah hanya {persen_valid:.1f}% valid — geser posisi kamera")
        return None

    zref = float(np.nanmedian(area_depth[area_mask]))
    return zref, r1, r2, c1, c2


def diagnostik_disparity_zref(disp_float, h, w, crop, Q=None):
    """
    DIAGNOSTIK: tampilkan statistik disparity MENTAH (sebelum filter Q,
    sebelum filter rentang) khusus di area tengah frame (area Z_ref).
    Tujuan: membuktikan apakah disparity di sana memang konsisten kecil
    (salah), atau bervariasi tapi sebagian besar tertolak filter rentang.
    """
    r1 = int(h * (0.5 - crop/2))
    r2 = int(h * (0.5 + crop/2))
    c1 = int(w * (0.5 - crop/2))
    c2 = int(w * (0.5 + crop/2))

    area_disp = disp_float[r1:r2, c1:c2]
    disp_masuk_akal = area_disp[area_disp > -1000]  # buang nilai -2048

    gagal_total = int((area_disp <= -1000).sum())
    print(f"\n[DIAGNOSTIK ZREF] Area tengah ({crop*100:.0f}% frame):")
    print(f"  Piksel gagal match (~-2048): {gagal_total} dari {area_disp.size} "
          f"({100*gagal_total/area_disp.size:.1f}%)")

    if disp_masuk_akal.size > 0:
        pct = np.percentile(disp_masuk_akal, [5, 25, 50, 75, 95])
        print(f"  Disparity valid -> count={disp_masuk_akal.size} "
              f"p5={pct[0]:.1f} p25={pct[1]:.1f} p50={pct[2]:.1f} p75={pct[3]:.1f} p95={pct[4]:.1f}")

        # Hitung disparity yang seharusnya berdasarkan Q aktif.
        focal, baseline = info_q(Q)
        if focal is not None and baseline is not None:
            disp_seharusnya_50cm = (focal * baseline) / 0.50
            disp_seharusnya_60cm = (focal * baseline) / 0.60
            print(f"  Untuk depth fisik 50-60cm, disparity SEHARUSNYA berkisar "
                  f"{disp_seharusnya_60cm:.1f} - {disp_seharusnya_50cm:.1f} px")
            print(f"  Median disparity AKTUAL di area ini: {pct[2]:.1f} px")
            if pct[2] < disp_seharusnya_60cm * 0.5:
                print(f"  [KESIMPULAN] Disparity aktual JAUH lebih kecil dari seharusnya")
                print(f"               -> SGBM gagal mencocokkan dengan benar di area ini")
            else:
                print(f"  [KESIMPULAN] Disparity aktual mendekati rentang yang diharapkan")
        else:
            print("  [INFO] Focal/baseline dari Q tidak bisa dihitung, diagnostik rentang disparity dilewati.")


def simpan_zref(zref_m, image_size, n_frame_stabil=1, stats_roi=None):
    """Simpan nilai Z_ref ke file .npz"""
    path = os.path.join(OUTPUT_DIR, "zref.npz")
    zref_vertical_m = koreksi_tilt_z(zref_m)
    stats_roi = stats_roi or {}
    np.savez(path,
             zref_m        = np.array([zref_m]),
             zref_cm       = np.array([zref_m * 100]),
             zref_vertical_m = np.array([zref_vertical_m]),
             zref_vertical_cm = np.array([zref_vertical_m * 100]),
             image_size    = np.array(image_size),
             stabilisasi_aktif = np.array([int(STABILISASI_AKTIF)]),
             n_frame_stabil = np.array([int(n_frame_stabil)]),
             zref_roi_valid_percent = np.array([float(stats_roi.get("valid_roi", np.nan))]),
             zref_roi_std_cm = np.array([float(stats_roi.get("std_cm", np.nan))]),
             camera_height_cm_info = np.array([float(CAMERA_HEIGHT_CM)]))
    print(f"\n  [✓] Z_ref tersimpan → {path}")
    print(f"  [✓] Z_ref raw = {zref_m*100:.2f} cm = {zref_m:.4f} m")
    print(f"  [✓] Z_ref vert≈ {zref_vertical_m*100:.2f} cm")
    print(f"  [✓] Sumber Z_ref: median temporal {n_frame_stabil} frame")
    print(f"\n  Artinya: tinggi batubara = Z_ref - Z_batubara")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    snap_dir = os.path.join(OUTPUT_DIR, "snapshot_depth")
    os.makedirs(snap_dir, exist_ok=True)

    # ─── Muat stereo maps ───
    print("\n" + "="*55)
    print("  SCRIPT 5 — DEPTH MAP + Z_REF")
    print("="*55)
    maps = muat_stereomap()
    if maps is None:
        return

    Q = maps["Q"]
    image_size = maps.get("image_size", (maps["mapLx"].shape[1], maps["mapLx"].shape[0]))

    # FIX: pastikan ukuran target kalibrasi tersedia sebagai variabel eksplisit.
    # image_size berformat (width, height). Variabel w_t/h_t dipakai saat simpan Z_ref
    # dan diagnostik disparity. Tanpa ini, tombol Z/X bisa memunculkan NameError.
    if isinstance(image_size, np.ndarray):
        image_size = tuple(map(int, image_size.flatten()[:2]))
    else:
        image_size = tuple(map(int, image_size))
    w_t, h_t = image_size

    # ─── Buat SGBM ───
    left_matcher, right_matcher = buat_sgbm()
    print(f"  [✓] StereoSGBM siap (numDisp={SGBM_NUM_DISP}, blockSize={SGBM_BLOCK_SIZE})")

    # ─── Buka kamera ───
    cap_kiri  = buka_kamera(CAMERA_ID_KIRI, "kiri")
    cap_kanan = buka_kamera(CAMERA_ID_KANAN, "kanan")

    if cap_kiri is None or cap_kanan is None:
        print("[ERROR] Kamera tidak bisa dibuka!")
        return

    print(f"  [✓] Kamera dibuka raw: {CAMERA_RAW_WIDTH}x{CAMERA_RAW_HEIGHT} | target kalibrasi: {image_size}")
    print(f"  [✓] Crop realtime: {TERAPKAN_CROP_REALTIME} | ROI kiri: {ROI_LEFT} | ROI kanan: {ROI_RIGHT}")

    print(f"\n  KONTROL:")
    print(f"  [D]   → Ganti tampilan (depth colormap / frame asli / disparity raw)")
    print(f"  [Z]   → Mode Z_REF — arahkan ke gerbong KOSONG lalu tekan Z lagi")
    print(f"  [S]   → Simpan snapshot depth map STABIL dari median beberapa frame")
    print(f"  [M]   → Ganti jenis snapshot: LANTAI KOSONG / OBJEK BALOK / MUATAN BATUBARA")
    print(f"  [F]   → Reset buffer stabilisasi frame")
    print(f"  [X]   → DIAGNOSTIK: cetak statistik disparity mentah di area tengah")
    print(f"  [ESC] → Keluar\n")

    mode_tampil  = 0        # 0=depth, 1=frame asli, 2=disparity raw
    mode_zref    = False    # True = sedang dalam mode pengukuran Z_ref
    snap_counter = 0
    zref_aktif   = None     # nilai Z_ref yang sudah diukur
    snapshot_mode_idx = 0   # 0=LANTAI_KOSONG, 1=OBJEK_BALOK, 2=MUATAN_BATUBARA

    depth_buffer = deque(maxlen=STABILISASI_N_FRAME)
    median_roi_history = deque(maxlen=STABILISASI_N_FRAME)
    stabil_ok = False
    stabil_reason = "buffer kosong"

    # Cek apakah Z_ref sudah pernah disimpan
    path_zref = os.path.join(OUTPUT_DIR, "zref.npz")
    if os.path.exists(path_zref):
        data_zref = np.load(path_zref)
        zref_aktif = float(data_zref["zref_m"][0])
        print(f"  [INFO] Z_ref sebelumnya ditemukan: {zref_aktif*100:.2f} cm (stereo/raw)")
        if "zref_vertical_m" in data_zref:
            print(f"  [INFO] Z_ref vertikal tersimpan: {float(data_zref['zref_vertical_m'][0])*100:.2f} cm")

    label_mode = ["DEPTH MAP", "FRAME ASLI", "DISPARITY RAW"]
    WINDOW_TITLE = "Script 5 - Depth Map [D/Z/S/M/X/ESC]"
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, DISPLAY_CANVAS_WIDTH, DISPLAY_CANVAS_HEIGHT)

    while True:
        ret_kiri,  frame_kiri  = cap_kiri.read()
        ret_kanan, frame_kanan = cap_kanan.read()

        if not ret_kiri or not ret_kanan:
            print("[ERROR] Gagal membaca frame kamera")
            break

        # ─── 1. Samakan format frame live dengan format kalibrasi Script 06 ───
        frame_kiri = persiapkan_frame_realtime(frame_kiri, "kiri", image_size)
        frame_kanan = persiapkan_frame_realtime(frame_kanan, "kanan", image_size)
        frame_kiri = pastikan_ukuran_frame(frame_kiri, image_size, "kiri")
        frame_kanan = pastikan_ukuran_frame(frame_kanan, image_size, "kanan")

        # ─── 2. Remap (undistort + rektifikasi) ───
        remap_kiri  = cv2.remap(frame_kiri,  maps["mapLx"], maps["mapLy"], cv2.INTER_LINEAR)
        remap_kanan = cv2.remap(frame_kanan, maps["mapRx"], maps["mapRy"], cv2.INTER_LINEAR)

        gray_kiri  = cv2.cvtColor(remap_kiri,  cv2.COLOR_BGR2GRAY)
        gray_kanan = cv2.cvtColor(remap_kanan, cv2.COLOR_BGR2GRAY)

        # ─── 2. Disparity map (SGBM + WLS) ───
        disp_float = hitung_disparity_wls(gray_kiri, gray_kanan, left_matcher, right_matcher)

        # ─── 3. Konversi ke depth map ───
        depth_map, valid_mask = disp_ke_depth(disp_float, Q)

        # ─── 4. Median filter ───
        depth_filtered = median_filter(depth_map, kernel=5)
        valid_mask_clean = ~np.isnan(depth_filtered)

        # ─── 5. Stabilisasi temporal beberapa frame ───
        depth_buffer.append(depth_filtered.copy())

        stats_frame = statistik_roi_depth(depth_filtered, valid_mask_clean, ZREF_CENTER_CROP)
        if np.isfinite(stats_frame["median_cm"]):
            median_roi_history.append(float(stats_frame["median_cm"]))

        depth_stabil, valid_mask_stabil = hitung_depth_temporal_median(depth_buffer)
        if depth_stabil is not None:
            depth_pakai = depth_stabil
            valid_mask_pakai = valid_mask_stabil
            sumber_depth = f"median {len(depth_buffer)} frame"
        else:
            depth_pakai = depth_filtered
            valid_mask_pakai = valid_mask_clean
            sumber_depth = f"single frame ({len(depth_buffer)}/{STABILISASI_MIN_FRAME})"

        # ─── 6. Hitung persen valid + status stabil ───
        persen_valid = hitung_persen_valid(valid_mask_pakai)
        persen_valid_roi = hitung_persen_valid_area_tengah(valid_mask_pakai, ZREF_CENTER_CROP)
        stats_roi = statistik_roi_depth(depth_pakai, valid_mask_pakai, ZREF_CENTER_CROP)
        stabil_ok, stabil_reason = status_stabil(stats_roi, median_roi_history, len(depth_buffer))

        # ─── Buat tampilan ───
        if mode_tampil == 0:
            tampilan = depth_ke_colormap(depth_pakai, valid_mask_pakai)
        elif mode_tampil == 1:
            tampilan = remap_kiri.copy()
        else:
            # Disparity raw — normalisasi ke 0-255
            disp_norm = cv2.normalize(disp_float, None, 0, 255, cv2.NORM_MINMAX)
            disp_norm = np.uint8(np.clip(disp_norm, 0, 255))
            tampilan  = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

        # ─── Dashboard tampilan penuh: teks di panel atas, gambar depth/frame di bawah ───
        hasil_zref_display = None
        if mode_zref:
            hasil_zref_display = ukur_zref(depth_pakai, valid_mask_pakai)

        tampilan_display = buat_dashboard_display(
            tampilan=tampilan,
            label_mode_text=label_mode[mode_tampil],
            sumber_depth=sumber_depth,
            persen_valid=persen_valid,
            persen_valid_roi=persen_valid_roi,
            stabil_ok=stabil_ok,
            stabil_reason=stabil_reason,
            zref_aktif=zref_aktif,
            mode_zref=mode_zref,
            hasil_zref=hasil_zref_display,
            stats_roi=stats_roi,
            snapshot_label=SNAPSHOT_MODES[snapshot_mode_idx][1],
            snapshot_code=SNAPSHOT_MODES[snapshot_mode_idx][0],
        )

        cv2.imshow(WINDOW_TITLE, tampilan_display)

        # ─── Keyboard ───
        key = cv2.waitKey(1) & 0xFF

        if key == 27:   # ESC
            break

        elif key == ord('d') or key == ord('D'):
            mode_tampil = (mode_tampil + 1) % 3
            print(f"  [D] Tampilan: {label_mode[mode_tampil]}")

        elif key == ord('z') or key == ord('Z'):
            if not mode_zref:
                mode_zref = True
                print(f"\n  [Z] MODE Z_REF AKTIF")
                print(f"      Pastikan gerbong KOSONG (tanpa balok, tanpa batubara, tanpa objek)")
                print(f"      Snapshot mode saat ini: {SNAPSHOT_MODES[snapshot_mode_idx][1]} (tidak memengaruhi Z_ref)")
                print(f"      Tekan [Z] lagi untuk menyimpan nilai Z_ref")
            else:
                # Simpan Z_ref
                hasil_zref = ukur_zref(depth_pakai, valid_mask_pakai)
                if hasil_zref is not None:
                    if not stabil_ok:
                        print(f"  [WARNING] Z_ref disimpan saat status belum stabil: {stabil_reason}")
                        print("            Sebaiknya tunggu indikator 'Stabil: YA' agar Z_ref lebih kuat.")
                        if WAJIB_STABIL_UNTUK_SIMPAN:
                            print("            Z_ref DITOLAK karena WAJIB_STABIL_UNTUK_SIMPAN=True")
                            continue
                    zref_aktif = hasil_zref[0]
                    simpan_zref(zref_aktif, (w_t, h_t), n_frame_stabil=len(depth_buffer), stats_roi=stats_roi)
                    mode_zref = False
                else:
                    print("  [WARNING] Z_ref gagal diukur — data tidak cukup valid")

        elif key == ord('s') or key == ord('S'):
            if (WAJIB_STABIL_UNTUK_SIMPAN and not stabil_ok):
                print(f"  [S] Snapshot DITOLAK — depth belum stabil: {stabil_reason}")
                continue
            if not stabil_ok:
                print(f"  [WARNING] Snapshot disimpan, tetapi status belum stabil: {stabil_reason}")

            snapshot_code, snapshot_label, snapshot_info = SNAPSHOT_MODES[snapshot_mode_idx]
            if snapshot_code == "LANTAI_KOSONG" and zref_aktif is None:
                print("  [WARNING] Snapshot lantai disimpan sebelum Z_ref tersimpan. Sebaiknya ukur Z_ref dulu dengan gerbong kosong.")
            if snapshot_code == "OBJEK_BALOK" and zref_aktif is None:
                print("  [WARNING] Snapshot objek/balok disimpan tanpa Z_ref. Tinggi objek di 05B bisa tidak valid.")

            snap_counter += 1
            path_snap = os.path.join(snap_dir, f"depth_snap_{snap_counter:03d}.npz")
            path_img  = os.path.join(snap_dir, f"depth_snap_{snap_counter:03d}.png")
            np.savez(path_snap,
                     snapshot_type    = np.array([snapshot_code]),
                     snapshot_label   = np.array([snapshot_label]),
                     snapshot_info    = np.array([snapshot_info]),
                     depth_map        = depth_pakai,
                     valid_mask       = valid_mask_pakai,
                     persen_valid     = persen_valid,
                     persen_valid_roi = persen_valid_roi,
                     zref             = np.array([zref_aktif]) if zref_aktif else np.array([np.nan]),
                     zref_vertical    = np.array([koreksi_tilt_z(zref_aktif)]) if zref_aktif else np.array([np.nan]),
                     stabilisasi_aktif = np.array([int(STABILISASI_AKTIF)]),
                     n_frame_stabil = np.array([len(depth_buffer)]),
                     stabil_ok = np.array([int(stabil_ok)]),
                     stabil_reason = np.array([stabil_reason]),
                     roi_median_cm = np.array([float(stats_roi.get("median_cm", np.nan))]),
                     roi_std_cm = np.array([float(stats_roi.get("std_cm", np.nan))]))
            cv2.imwrite(path_img, tampilan_display)
            print(f"  [S] Snapshot tersimpan → {path_snap}")
            print(f"      Jenis: {snapshot_label} ({snapshot_code})")
            print(f"      Sumber: {sumber_depth} | Stabil: {'YA' if stabil_ok else 'BELUM'}")
            print(f"      Piksel valid total: {persen_valid:.1f}% | ROI: {persen_valid_roi:.1f}% | {stabil_reason}")

        elif key == ord('m') or key == ord('M'):
            snapshot_mode_idx = (snapshot_mode_idx + 1) % len(SNAPSHOT_MODES)
            kode, label, info = SNAPSHOT_MODES[snapshot_mode_idx]
            depth_buffer.clear()
            median_roi_history.clear()
            mode_zref = False
            print(f"  [M] Jenis snapshot diganti: {label} ({kode})")
            print(f"      {info}")
            print("      Buffer direset agar frame lama tidak tercampur. Tunggu sampai Stabil: YA.")

        elif key == ord('f') or key == ord('F'):
            depth_buffer.clear()
            median_roi_history.clear()
            print("  [F] Buffer stabilisasi direset. Tunggu beberapa frame sebelum simpan lagi.")

        elif key == ord('x') or key == ord('X'):
            diagnostik_disparity_zref(disp_float, h_t, w_t, ZREF_CENTER_CROP, Q)

    cap_kiri.release()
    cap_kanan.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*55}")
    print(f"  Script 5 selesai.")
    if zref_aktif:
        print(f"  Z_ref tersimpan: {zref_aktif*100:.2f} cm")
        print(f"\n  LANGKAH BERIKUTNYA:")
        print(f"  → Akuisisi dataset batubara untuk training U-Net")
        print(f"  → Atau langsung: python 06_estimasi_volume.py (setelah U-Net siap)")
    else:
        print(f"  [!] Z_ref BELUM disimpan — jalankan lagi dan tekan [Z]")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
