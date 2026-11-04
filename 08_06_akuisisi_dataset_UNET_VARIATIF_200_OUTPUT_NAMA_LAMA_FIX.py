"""
SCRIPT 6 — AKUISISI DATASET U-NET VARIATIF 200 CITRA + OUTPUT NAMA LAMA
================================================================

Fungsi:
1. Mengambil citra kamera kiri untuk dataset U-Net.
2. Menyamakan alur input dengan pipeline kalibrasi yang memakai crop Script 06:
   kamera live -> crop ROI yang sama -> resize ke ukuran stereomap -> cv2.remap.
3. Membaca hasil_kalibrasi/stereomap.xml.
4. Mengikuti rencana dataset variatif 200 citra agar data tidak monoton.
5. Tetap menyimpan output ke folder lama dataset_unet/ agar script berikutnya tidak perlu diubah.
6. Menyimpan:
   - kiri_raw       : frame input yang sudah SELARAS dengan stereomap
                     (jika crop aktif, ini adalah crop-resize sebelum rectification)
   - kiri_rectified : frame setelah cv2.remap, ini yang disarankan untuk training U-Net
   - preview        : perbandingan input vs rectified
   - log_akuisisi.csv

Catatan penting:
- Jika kalibrasi/capture ChArUco sebelumnya memakai SIMPAN_FRAME_CROP=True,
  maka frame live pada script ini juga wajib di-crop dengan ROI yang sama sebelum cv2.remap.
- Dataset U-Net sebaiknya memakai folder kiri_rectified, karena tahap inference nanti juga
  memakai frame kiri yang sudah direktifikasi/remap.

Kontrol:
[S]   Simpan input + rectified + preview
[C]   Ganti tahap muatan dan variasi
[B]   Ganti label baseline
[G]   Tampilkan/sembunyikan garis panduan ROI
[ESC] Keluar
"""

import cv2
import os
import csv
import re
import numpy as np
from datetime import datetime

# =========================================================
# KONFIGURASI FOLDER
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KALIBRASI_DIR = os.path.join(BASE_DIR, "hasil_kalibrasi")
STEREOMAP_XML = os.path.join(KALIBRASI_DIR, "stereomap.xml")

DATASET_ROOT = os.path.join(BASE_DIR, "dataset_unet")
OUTPUT_RAW_ROOT = os.path.join(DATASET_ROOT, "kiri_raw")
OUTPUT_RECTIFIED_ROOT = os.path.join(DATASET_ROOT, "kiri_rectified")
OUTPUT_PREVIEW_ROOT = os.path.join(DATASET_ROOT, "preview_rektifikasi")
OUTPUT_CAMERA_RAW_ROOT = os.path.join(DATASET_ROOT, "kiri_camera_raw_original")
LOG_CSV = os.path.join(DATASET_ROOT, "log_akuisisi.csv")
RENCANA_CSV = os.path.join(DATASET_ROOT, "rencana_akuisisi_dataset_200.csv")

# CATATAN PENTING:
# Folder output sengaja tetap memakai nama lama: dataset_unet/
# Tujuannya agar alur setelah akuisisi tidak perlu mengubah nama/path.
# Gambar yang disarankan untuk anotasi Roboflow tetap: dataset_unet/kiri_rectified/

# =========================================================
# KONFIGURASI KAMERA DAN CROP SELARAS SCRIPT 06
# =========================================================
CAMERA_ID_KIRI = 0
AUTO_DETECT_CAMERA = True
SCAN_CAMERA_MAX_ID = 10

# Ukuran raw kamera sebelum crop. Pada setup Anda umumnya 640x480.
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

# Harus True jika dataset kalibrasi sebelumnya berasal dari Script 06 dengan SIMPAN_FRAME_CROP=True.
TERAPKAN_CROP_REALTIME = True

# ROI sama dengan Script 06/Script 4/Script 5 yang sudah diselaraskan.
# Format: (x_awal, y_awal, x_akhir, y_akhir), skala 0.0 sampai 1.0
ROI_LEFT = (0.00, 0.05, 1.00, 0.95)

# Jika ukuran hasil crop berbeda dari stereomap, hasil crop akan di-resize ke image_size stereomap.
ALLOW_RESIZE_IF_NEEDED = True

# Simpan raw kamera asli sebelum crop. Default False agar dataset tidak terlalu banyak file.
SIMPAN_CAMERA_RAW_ORIGINAL = False

# Tampilan layar
DISPLAY_MAX_WIDTH = 1180
DISPLAY_MAX_HEIGHT = 680
TAMPILKAN_GARIS_ROI = True
DISPLAY_MIRROR_HORIZONTAL = True

# Stereomap wajib agar dataset rectified benar-benar selaras.
REQUIRE_STEREOMAP = True
RMS_STEREO_BATAS_AMAN = 1.0

IMAGE_EXT = ".png"

# Target jumlah citra dibuat lebih variatif sesuai rancangan 200 citra.
# Prinsip rancangan:
# - Dataset tidak didominasi kondisi kosong, merata, atau gunungan/overload.
# - Variasi diperbanyak pada muatan kecil sampai tinggi karena bentuk mask berubah banyak.
# - Kondisi penuh datar diasumsikan sekitar 45 genggam berdasarkan uji awal.
# - Istilah genggam hanya untuk panduan akuisisi. Jika memungkinkan, setiap tahap tetap ditimbang dalam gram.
TARGET = {
    "kosong": {
        "referensi_gerbong_kosong": 8,
    },

    "sangat_sedikit_1_6_butir": {
        "1_butir_tengah": 2,
        "1_butir_kiri": 2,
        "1_butir_kanan": 2,
        "2_3_butir_depan": 2,
        "2_3_butir_belakang": 2,
        "4_6_butir_acak_asimetris": 2,
    },

    "sedikit_butiran_7_10_butir": {
        "menyebar_sebagian": 3,
        "kiri_dominan": 3,
        "kanan_dominan": 3,
        "depan_dominan": 3,
        "belakang_dominan": 3,
    },

    "awal_pengisian_1_5_genggam": {
        "1_genggam_tengah": 3,
        "2_genggam_kiri": 3,
        "3_genggam_kanan": 3,
        "4_genggam_depan": 3,
        "5_genggam_belakang": 3,
    },

    "rendah_6_10_genggam": {
        "6_genggam_kiri": 3,
        "7_genggam_kanan": 3,
        "8_genggam_depan": 3,
        "9_genggam_belakang": 3,
        "10_genggam_asimetris": 3,
        "10_genggam_memanjang": 3,
    },

    "rendah_11_15_genggam": {
        "11_genggam_kiri_dominan": 3,
        "12_genggam_kanan_dominan": 3,
        "13_genggam_depan_belakang": 3,
        "14_genggam_diagonal": 3,
        "15_genggam_asimetris": 3,
        "15_genggam_menyebar_tidak_merata": 3,
    },

    "sedang_16_20_genggam": {
        "16_genggam_rendah_miring": 3,
        "17_genggam_kiri_dominan": 3,
        "18_genggam_kanan_dominan": 3,
        "19_genggam_depan_dominan": 3,
        "20_genggam_belakang_dominan": 3,
        "20_genggam_gundukan_kecil": 3,
    },

    "sedang_21_25_genggam": {
        "21_genggam_diagonal": 3,
        "22_genggam_memanjang": 3,
        "23_genggam_tengah_dominan": 3,
        "24_genggam_depan_belakang": 3,
        "25_genggam_asimetris": 3,
        "25_genggam_dua_area": 3,
    },

    "sedang_26_30_genggam": {
        "26_genggam_rata_sebagian": 3,
        "27_genggam_satu_sisi_dominan": 3,
        "28_genggam_dua_gundukan_kecil": 3,
        "29_genggam_memanjang": 3,
        "30_genggam_asimetris": 3,
        "30_genggam_tengah_dominan": 3,
    },

    "tinggi_31_35_genggam": {
        "31_genggam_hampir_menutup_lantai": 3,
        "32_genggam_kiri_dominan": 3,
        "33_genggam_kanan_dominan": 3,
        "34_genggam_memanjang": 3,
        "35_genggam_asimetris": 3,
        "35_genggam_gundukan_sedang": 3,
    },

    "tinggi_36_40_genggam": {
        "36_genggam_sisa_ruang_depan": 4,
        "37_38_genggam_sisa_ruang_belakang": 4,
        "39_genggam_sisa_ruang_samping": 4,
        "40_genggam_hampir_penuh_tidak_merata": 4,
    },

    "penuh_datar_41_45_genggam": {
        "41_42_genggam_penuh_landai": 3,
        "43_genggam_penuh_asimetris": 3,
        "44_genggam_penuh_hampir_datar": 4,
        "45_genggam_penuh_datar": 4,
    },

    "overload_ringan_diatas_45_genggam": {
        "heaping_ringan_tengah": 4,
        "heaping_ringan_memanjang": 4,
        "heaping_ringan_asimetris": 4,
    },
}


# =========================================================
# RENCANA AKUISISI DATASET
# =========================================================
def total_target_dataset():
    """Menghitung total target citra dari struktur TARGET."""
    total = 0
    for profil_dict in TARGET.values():
        for jumlah in profil_dict.values():
            total += int(jumlah)
    return total


def tulis_rencana_akuisisi_csv():
    """
    Menulis rencana akuisisi dataset ke CSV agar operator tahu urutan,
    kondisi muatan, variasi/profil, dan jumlah target setiap variasi.
    """
    os.makedirs(DATASET_ROOT, exist_ok=True)

    with open(RENCANA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "no",
            "kondisi_muatan",
            "profil_permukaan",
            "target_citra",
            "folder_kiri_raw",
            "folder_kiri_rectified",
            "folder_preview",
            "catatan",
        ])

        no = 1
        for kondisi, profil_dict in TARGET.items():
            for profil, target_jumlah in profil_dict.items():
                writer.writerow([
                    no,
                    kondisi,
                    profil,
                    int(target_jumlah),
                    os.path.join(OUTPUT_RAW_ROOT, kondisi, profil),
                    os.path.join(OUTPUT_RECTIFIED_ROOT, kondisi, profil),
                    os.path.join(OUTPUT_PREVIEW_ROOT, kondisi, profil),
                    "Gunakan kiri_rectified untuk anotasi/training U-Net",
                ])
                no += 1

    print(f"[OK] Rencana akuisisi dataset ditulis: {RENCANA_CSV}")
    print(f"     Total target: {total_target_dataset()} citra")

# =========================================================
# UTILITAS UMUM
# =========================================================
def token_aman(teks):
    teks = str(teks).strip().lower()
    teks = teks.replace(" ", "_").replace("-", "_").replace(".", "_")
    teks = re.sub(r"[^a-zA-Z0-9_]", "_", teks)
    teks = re.sub(r"_+", "_", teks)
    return teks.strip("_") or "unknown"


def _node_empty(node):
    try:
        return node.empty()
    except Exception:
        return True


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


# =========================================================
# STEREOMAP
# =========================================================
def muat_stereomap_kiri(path_xml):
    if not os.path.exists(path_xml):
        if REQUIRE_STEREOMAP:
            raise FileNotFoundError(
                f"stereomap.xml tidak ditemukan: {path_xml}\n"
                "Jalankan dulu Script 3 dan Script 4 sampai menghasilkan stereomap.xml yang valid."
            )
        print("[WARNING] stereomap.xml tidak ditemukan. Hanya input image yang akan dipakai.")
        return None

    fs = cv2.FileStorage(path_xml, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Gagal membuka XML: {path_xml}")

    mapLx = baca_matrix(fs, "stereoMapL_x")
    mapLy = baca_matrix(fs, "stereoMapL_y")

    image_width = baca_int(fs, "image_width", default=mapLx.shape[1])
    image_height = baca_int(fs, "image_height", default=mapLx.shape[0])

    Q = baca_matrix(fs, "Q", wajib=False)
    R = baca_matrix(fs, "R", wajib=False)
    T = baca_matrix(fs, "T", wajib=False)
    rms_stereo = baca_float(fs, "rms_stereo", default=None)
    baseline_mm = baca_float(fs, "baseline_mm", default=None)
    valid_pairs = baca_int(fs, "valid_pairs", default=None)

    fs.release()

    print("[OK] stereomap.xml berhasil dimuat")
    print(f"     File         : {path_xml}")
    print(f"     Image size   : {(image_width, image_height)}")
    print(f"     Map kiri     : {mapLx.shape} / {mapLy.shape}")
    print(f"     Q matrix     : {'tersedia' if Q is not None else 'tidak tersedia'}")
    print(f"     R/T          : {'tersedia' if R is not None and T is not None else 'tidak tersedia'}")
    if rms_stereo is not None:
        print(f"     RMS stereo   : {rms_stereo:.6f} px")
        if rms_stereo > RMS_STEREO_BATAS_AMAN:
            print("     [WARNING] RMS stereo > 1 px. Dataset masih bisa diambil, tetapi cek lagi kualitas rectification.")
    if baseline_mm is not None:
        print(f"     Baseline     : {baseline_mm:.3f} mm | {baseline_mm/10.0:.3f} cm")
    if valid_pairs is not None:
        print(f"     Valid pairs  : {valid_pairs}")

    return {
        "mapLx": mapLx,
        "mapLy": mapLy,
        "image_size": (image_width, image_height),
        "Q": Q,
        "R": R,
        "T": T,
        "rms_stereo": rms_stereo,
        "baseline_mm": baseline_mm,
        "valid_pairs": valid_pairs,
    }


# =========================================================
# ROI / CROP
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


def pastikan_ukuran_frame(frame, image_size):
    target_w, target_h = image_size
    h, w = frame.shape[:2]
    if (w, h) == image_size:
        return frame

    if not ALLOW_RESIZE_IF_NEEDED:
        raise RuntimeError(
            f"Ukuran frame {(w, h)} berbeda dari ukuran stereomap {image_size}. "
            "Atur resolusi/crop agar sama."
        )

    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def siapkan_frame_input(frame_camera_raw, image_size):
    if TERAPKAN_CROP_REALTIME:
        return crop_resize_frame(frame_camera_raw, ROI_LEFT, image_size)
    return pastikan_ukuran_frame(frame_camera_raw, image_size)


# =========================================================
# STRUKTUR DATASET DAN LOG
# =========================================================
def buat_struktur_folder():
    roots = [OUTPUT_RAW_ROOT, OUTPUT_RECTIFIED_ROOT, OUTPUT_PREVIEW_ROOT]
    if SIMPAN_CAMERA_RAW_ORIGINAL:
        roots.append(OUTPUT_CAMERA_RAW_ROOT)

    for root in roots:
        for kondisi, profil_dict in TARGET.items():
            for profil in profil_dict:
                os.makedirs(os.path.join(root, kondisi, profil), exist_ok=True)

    os.makedirs(DATASET_ROOT, exist_ok=True)


def path_folder_output(root, kondisi, profil):
    return os.path.join(root, kondisi, profil)


def hitung_existing(kondisi, profil, baseline_label):
    baseline_token = token_aman(baseline_label)
    kondisi_token = token_aman(kondisi)
    profil_token = token_aman(profil)
    prefix = f"{baseline_token}_{kondisi_token}_{profil_token}_"

    folder = path_folder_output(OUTPUT_RECTIFIED_ROOT, kondisi, profil)
    if not os.path.exists(folder):
        return 0

    nomor_tertinggi = 0
    for nama in os.listdir(folder):
        if not nama.startswith(prefix):
            continue
        match = re.search(r"_(\d{3,6})_", nama)
        if match:
            nomor_tertinggi = max(nomor_tertinggi, int(match.group(1)))
    return nomor_tertinggi


def tulis_log(
    baseline_label,
    kondisi,
    profil,
    nomor,
    filename_raw,
    path_raw,
    filename_rectified,
    path_rectified,
    filename_preview,
    path_preview,
    image_size,
    menggunakan_stereomap,
    crop_aktif,
    roi_left,
    rms_stereo,
    baseline_mm,
    camera_raw_path=""
):
    file_exists = os.path.exists(LOG_CSV)

    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "baseline_label",
                "kondisi_muatan",
                "profil_permukaan",
                "nomor",
                "filename_raw_input",
                "path_raw_input",
                "filename_rectified",
                "path_rectified",
                "filename_preview",
                "path_preview",
                "image_width",
                "image_height",
                "menggunakan_stereomap",
                "crop_aktif",
                "roi_left",
                "rms_stereo",
                "baseline_mm",
                "camera_raw_original_path",
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            baseline_label,
            kondisi,
            profil,
            nomor,
            filename_raw,
            path_raw,
            filename_rectified,
            path_rectified,
            filename_preview,
            path_preview,
            image_size[0],
            image_size[1],
            menggunakan_stereomap,
            crop_aktif,
            str(tuple(roi_left)),
            "" if rms_stereo is None else f"{rms_stereo:.6f}",
            "" if baseline_mm is None else f"{baseline_mm:.6f}",
            camera_raw_path,
        ])


# =========================================================
# MENU
# =========================================================
def tampilkan_menu_kondisi():
    print("\n  ── PILIH TAHAP MUATAN ──")
    daftar = list(TARGET.keys())
    for i, k in enumerate(daftar):
        print(f"    [{i+1}] {k}")
    return daftar


def tampilkan_menu_profil(kondisi):
    print(f"\n  ── PILIH VARIASI POSISI / PROFIL (kondisi: {kondisi}) ──")
    daftar = list(TARGET[kondisi].keys())
    for i, p in enumerate(daftar):
        target = TARGET[kondisi][p]
        print(f"    [{i+1}] {p}  (target: {target} citra)")
    return daftar


def baca_pilihan_int(prompt, jumlah):
    while True:
        try:
            nilai = int(input(prompt))
            if 1 <= nilai <= jumlah:
                return nilai - 1
        except Exception:
            pass
        print(f"  Pilihan tidak valid. Masukkan angka 1 sampai {jumlah}.")


def pilih_kondisi_profil():
    daftar_kondisi = tampilkan_menu_kondisi()
    idx_kondisi = baca_pilihan_int("\n  Pilih nomor tahap muatan: ", len(daftar_kondisi))
    kondisi = daftar_kondisi[idx_kondisi]

    daftar_profil = tampilkan_menu_profil(kondisi)
    idx_profil = baca_pilihan_int("\n  Pilih nomor variasi: ", len(daftar_profil))
    profil = daftar_profil[idx_profil]
    return kondisi, profil


# =========================================================
# VISUALISASI
# =========================================================
def label_gambar(img, teks1, teks2=None, warna=(0, 255, 0)):
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 72), (0, 0, 0), -1)
    cv2.putText(out, teks1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, warna, 2)
    if teks2:
        cv2.putText(out, teks2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    return out


def mirror_untuk_tampilan(img):
    return cv2.flip(img, 1) if DISPLAY_MIRROR_HORIZONTAL else img


def gambar_garis_roi(frame):
    out = frame.copy()
    if not TAMPILKAN_GARIS_ROI:
        return out

    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 72), (w - 1, h - 1), (0, 255, 255), 1)
    cv2.putText(out, "INPUT SELARAS STEREOMAP", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
    return out


def buat_preview(frame_input, frame_rectified, baseline_label, kondisi, profil, counter, target_now, rms_stereo=None):
    rms_text = "" if rms_stereo is None else f" | RMS={rms_stereo:.3f}px"
    crop_text = "CROP ON" if TERAPKAN_CROP_REALTIME else "CROP OFF"

    raw_label = label_gambar(
        gambar_garis_roi(mirror_untuk_tampilan(frame_input)),
        "KIRI INPUT - SEBELUM REKTIFIKASI",
        f"{crop_text} | Baseline={baseline_label} | {kondisi}/{profil}{rms_text}",
        warna=(0, 180, 255)
    )

    rect_label = label_gambar(
        mirror_untuk_tampilan(frame_rectified),
        "KIRI RECTIFIED - SETELAH CV2.REMAP",
        f"Tersimpan={counter}/{target_now} | PAKAI INI UNTUK DATASET U-NET",
        warna=(0, 255, 0)
    )

    return np.hstack([raw_label, rect_label])


def resize_agar_muat_layar(img):
    h, w = img.shape[:2]
    scale = min(DISPLAY_MAX_WIDTH / w, DISPLAY_MAX_HEIGHT / h, 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# =========================================================
# SIMPAN DATASET
# =========================================================
def simpan_frame_dataset(
    frame_camera_raw,
    frame_input,
    frame_rectified,
    preview,
    baseline_label,
    kondisi,
    profil,
    nomor,
    image_size,
    menggunakan_stereomap,
    rms_stereo,
    baseline_mm
):
    baseline_token = token_aman(baseline_label)
    kondisi_token = token_aman(kondisi)
    profil_token = token_aman(profil)
    base_name = f"{baseline_token}_{kondisi_token}_{profil_token}_{nomor:03d}"

    filename_raw = f"{base_name}_raw_input{IMAGE_EXT}"
    filename_rectified = f"{base_name}_rectified{IMAGE_EXT}"
    filename_preview = f"{base_name}_preview{IMAGE_EXT}"

    path_raw = os.path.join(path_folder_output(OUTPUT_RAW_ROOT, kondisi, profil), filename_raw)
    path_rectified = os.path.join(path_folder_output(OUTPUT_RECTIFIED_ROOT, kondisi, profil), filename_rectified)
    path_preview = os.path.join(path_folder_output(OUTPUT_PREVIEW_ROOT, kondisi, profil), filename_preview)

    camera_raw_path = ""
    if SIMPAN_CAMERA_RAW_ORIGINAL:
        filename_camera_raw = f"{base_name}_camera_raw_original{IMAGE_EXT}"
        camera_raw_path = os.path.join(path_folder_output(OUTPUT_CAMERA_RAW_ROOT, kondisi, profil), filename_camera_raw)
        cv2.imwrite(camera_raw_path, frame_camera_raw)

    cv2.imwrite(path_raw, frame_input)
    cv2.imwrite(path_rectified, frame_rectified)
    cv2.imwrite(path_preview, preview)

    tulis_log(
        baseline_label=baseline_label,
        kondisi=kondisi,
        profil=profil,
        nomor=nomor,
        filename_raw=filename_raw,
        path_raw=path_raw,
        filename_rectified=filename_rectified,
        path_rectified=path_rectified,
        filename_preview=filename_preview,
        path_preview=path_preview,
        image_size=image_size,
        menggunakan_stereomap=menggunakan_stereomap,
        crop_aktif=TERAPKAN_CROP_REALTIME,
        roi_left=ROI_LEFT,
        rms_stereo=rms_stereo,
        baseline_mm=baseline_mm,
        camera_raw_path=camera_raw_path,
    )

    print("\n[OK] Dataset tersimpan:")
    print(f"     INPUT     : {path_raw}")
    print(f"     RECTIFIED : {path_rectified}")
    print(f"     PREVIEW   : {path_preview}")
    if camera_raw_path:
        print(f"     RAW ASLI  : {camera_raw_path}")


# =========================================================
# KAMERA
# =========================================================
def daftar_backend():
    backends = [("DEFAULT", None)]

    if os.name == "nt":
        backends.extend([
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
        ])

    backends.append(("ANY", cv2.CAP_ANY))
    return backends


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
        set_property_aman(cap, cv2.CAP_PROP_FOCUS, FOCUS_LEFT if str(nama).upper() == "KIRI" else FOCUS_RIGHT)


def buka_kamera_default(camera_id, backend=None):
    if backend is None:
        cap = cv2.VideoCapture(camera_id)
    else:
        cap = cv2.VideoCapture(camera_id, backend)

    if cap.isOpened():
        set_property_aman(cap, cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        set_property_aman(cap, cv2.CAP_PROP_FRAME_WIDTH, int(CAMERA_RAW_WIDTH))
        set_property_aman(cap, cv2.CAP_PROP_FRAME_HEIGHT, int(CAMERA_RAW_HEIGHT))
        set_property_aman(cap, cv2.CAP_PROP_FPS, int(FPS))
        set_property_aman(cap, cv2.CAP_PROP_BUFFERSIZE, 1)
        samakan_setting_kamera(cap, "KIRI")
        return cap

    cap.release()
    return None


def baca_frame_awal(cap, jumlah=10):
    frame = None
    for _ in range(jumlah):
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
    return frame


def buka_kamera_kiri():
    print(f"\n[INFO] Membuka kamera kiri ID default: {CAMERA_ID_KIRI}")

    for nama_backend, backend in daftar_backend():
        cap = buka_kamera_default(CAMERA_ID_KIRI, backend)
        if cap is not None:
            frame = baca_frame_awal(cap)
            if frame is not None:
                print(f"[OK] Kamera kiri terbuka pada ID={CAMERA_ID_KIRI}, backend={nama_backend}")
                print(f"     Ukuran frame raw aktual: {(frame.shape[1], frame.shape[0])}")
                return cap
            cap.release()

    if not AUTO_DETECT_CAMERA:
        raise RuntimeError(f"Kamera kiri ID={CAMERA_ID_KIRI} tidak bisa dibuka.")

    print("[WARNING] Kamera default gagal. Mencari kamera otomatis...")
    for cam_id in range(SCAN_CAMERA_MAX_ID + 1):
        for nama_backend, backend in daftar_backend():
            cap = buka_kamera_default(cam_id, backend)
            if cap is not None:
                frame = baca_frame_awal(cap)
                if frame is not None:
                    print(f"[OK] Kamera kiri ditemukan pada ID={cam_id}, backend={nama_backend}")
                    print(f"     Ukuran frame raw aktual: {(frame.shape[1], frame.shape[0])}")
                    return cap
                cap.release()

    raise RuntimeError("Tidak ada kamera yang bisa dibuka. Cek koneksi kamera atau aplikasi lain yang memakai kamera.")


# =========================================================
# MAIN
# =========================================================
def main():
    buat_struktur_folder()
    tulis_rencana_akuisisi_csv()

    print("\n" + "=" * 78)
    print("SCRIPT 6 — AKUISISI DATASET U-NET VARIATIF 200 CITRA + OUTPUT NAMA LAMA")
    print("=" * 78)
    print(f"Stereomap XML      : {STEREOMAP_XML}")
    print(f"Output INPUT       : {OUTPUT_RAW_ROOT}")
    print(f"Output RECTIFIED   : {OUTPUT_RECTIFIED_ROOT}")
    print(f"Output PREVIEW     : {OUTPUT_PREVIEW_ROOT}")
    print(f"Log CSV            : {LOG_CSV}")
    print(f"Rencana CSV        : {RENCANA_CSV}")
    print(f"Total target       : {total_target_dataset()} citra")
    print(f"Crop realtime      : {TERAPKAN_CROP_REALTIME}")
    print(f"ROI kiri           : {ROI_LEFT}")
    print(f"Raw kamera target  : {CAMERA_RAW_WIDTH} x {CAMERA_RAW_HEIGHT} @ {FPS} FPS")
    print("=" * 78)

    stereomap = muat_stereomap_kiri(STEREOMAP_XML)
    menggunakan_stereomap = stereomap is not None

    if menggunakan_stereomap:
        image_size = stereomap["image_size"]
        mapLx = stereomap["mapLx"]
        mapLy = stereomap["mapLy"]
        rms_stereo = stereomap.get("rms_stereo")
        baseline_mm = stereomap.get("baseline_mm")
    else:
        image_size = (CAMERA_RAW_WIDTH, CAMERA_RAW_HEIGHT)
        mapLx = None
        mapLy = None
        rms_stereo = None
        baseline_mm = None

    print("\n[INFO] Dataset U-Net yang disarankan untuk training adalah folder:")
    print(f"       {OUTPUT_RECTIFIED_ROOT}")
    print("       Karena folder itu sudah mengikuti cv2.remap/stereomap.")

    baseline_label = input("\nMasukkan label baseline saat ini (contoh: bl10_07): ").strip()
    if not baseline_label:
        baseline_label = "bl_unknown"

    kondisi, profil = pilih_kondisi_profil()

    cap = buka_kamera_kiri()

    counter = hitung_existing(kondisi, profil, baseline_label)
    target_now = TARGET[kondisi][profil]

    print("\n" + "=" * 78)
    print(f"Mode aktif -> Baseline: {baseline_label} | Tahap: {kondisi} | Variasi: {profil}")
    print(f"Sudah ada  -> {counter} citra untuk kombinasi ini")
    print(f"Target     -> {target_now} citra")
    print("=" * 78)
    print("KONTROL:")
    print("[S]   Simpan input + rectified + preview")
    print("[C]   Ganti tahap muatan & variasi")
    print("[B]   Ganti baseline")
    print("[G]   Tampilkan/sembunyikan garis panduan")
    print("[ESC] Keluar")
    print("=" * 78)

    while True:
        ret, frame_camera_raw = cap.read()
        if not ret or frame_camera_raw is None:
            print("[ERROR] Gagal membaca frame kamera")
            break

        frame_input = siapkan_frame_input(frame_camera_raw, image_size)

        if menggunakan_stereomap:
            frame_rectified = cv2.remap(frame_input, mapLx, mapLy, cv2.INTER_LINEAR)
        else:
            frame_rectified = frame_input.copy()

        preview = buat_preview(
            frame_input,
            frame_rectified,
            baseline_label,
            kondisi,
            profil,
            counter,
            target_now,
            rms_stereo=rms_stereo
        )

        preview_tampil = resize_agar_muat_layar(preview)
        cv2.imshow("Akuisisi Dataset U-Net - INPUT vs RECTIFIED [S/C/B/G/ESC]", preview_tampil)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == ord("s") or key == ord("S"):
            counter += 1
            preview_save = buat_preview(
                frame_input,
                frame_rectified,
                baseline_label,
                kondisi,
                profil,
                counter,
                target_now,
                rms_stereo=rms_stereo
            )

            simpan_frame_dataset(
                frame_camera_raw=frame_camera_raw,
                frame_input=frame_input,
                frame_rectified=frame_rectified,
                preview=preview_save,
                baseline_label=baseline_label,
                kondisi=kondisi,
                profil=profil,
                nomor=counter,
                image_size=image_size,
                menggunakan_stereomap=menggunakan_stereomap,
                rms_stereo=rms_stereo,
                baseline_mm=baseline_mm,
            )

            status = "target tercapai" if counter >= target_now else f"perlu {target_now - counter} lagi"
            print(f"[INFO] Progress {counter}/{target_now} — {status}")

        elif key == ord("c") or key == ord("C"):
            cv2.destroyAllWindows()
            kondisi, profil = pilih_kondisi_profil()
            counter = hitung_existing(kondisi, profil, baseline_label)
            target_now = TARGET[kondisi][profil]
            print(f"\n[INFO] Mode baru -> Baseline={baseline_label} | Tahap={kondisi} | Variasi={profil} | Sudah ada={counter}")

        elif key == ord("b") or key == ord("B"):
            cv2.destroyAllWindows()
            baseline_label = input("\nMasukkan label baseline baru: ").strip()
            if not baseline_label:
                baseline_label = "bl_unknown"
            counter = hitung_existing(kondisi, profil, baseline_label)
            target_now = TARGET[kondisi][profil]
            print(f"\n[INFO] Baseline baru -> {baseline_label} | Sudah ada={counter}")

        elif key == ord("g") or key == ord("G"):
            global TAMPILKAN_GARIS_ROI
            TAMPILKAN_GARIS_ROI = not TAMPILKAN_GARIS_ROI
            print(f"[INFO] Garis panduan: {'ON' if TAMPILKAN_GARIS_ROI else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 78)
    print("SELESAI")
    print(f"Log dataset : {LOG_CSV}")
    print(f"INPUT       : {OUTPUT_RAW_ROOT}")
    print(f"RECTIFIED   : {OUTPUT_RECTIFIED_ROOT}")
    print(f"PREVIEW     : {OUTPUT_PREVIEW_ROOT}")
    print("\nLANGKAH LANJUTAN:")
    print("1. Gunakan folder kiri_rectified sebagai citra input U-Net.")
    print("2. Buat mask manual untuk setiap citra rectified.")
    print("3. Mask kosong = hitam semua; mask batubara = area batubara putih.")
    print("=" * 78)


if __name__ == "__main__":
    main()
