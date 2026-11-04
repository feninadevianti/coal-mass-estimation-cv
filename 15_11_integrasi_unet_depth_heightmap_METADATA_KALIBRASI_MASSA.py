"""
SCRIPT 11 — INTEGRASI U-NET MASK + DEPTH MAP + REFERENSI LANTAI LOKAL + METADATA KALIBRASI BERAT
VERSI REALTIME VALIDASI LANTAI 05B — AUTO FLOOR VISIBILITY UNTUK SEMUA KONDISI
====================================================

Fungsi utama:
1. Load model U-Net terbaik: hasil_training_unet_colab/model_best.pth atau hasil_training_unet/model_best.pth
2. Load stereomap: hasil_kalibrasi/stereomap.xml
3. Load Z_ref: hasil_kalibrasi/zref.xml atau hasil_kalibrasi/zref.npz
4. Buka kamera stereo kiri-kanan
5. Rectification dengan cv2.remap
6. Hitung disparity map dengan StereoSGBM
7. Konversi disparity ke depth map dengan Q matrix
8. Prediksi mask batubara dengan U-Net
9. Ambil depth hanya pada area mask batubara
10. Hitung height map utama: h(x,y) = depth_lantai_kosong(x,y) - depth_batubara(x,y)
11. Tampilkan realtime dan simpan snapshot jika tombol S ditekan

Catatan:
- Script ini menghitung HEIGHT MAP / peta tinggi batubara.
- Versi ini diselaraskan dengan Script 05B terbaru: jika tersedia snapshot LANTAI_KOSONG dari Script 05, tinggi dihitung memakai floor reference map lokal per piksel, bukan hanya satu angka Z_ref global.
- Volume final belum dihitung otomatis karena membutuhkan konversi luas piksel ke luas dunia nyata.
- Output height_m dapat dipakai untuk tahap volume berikutnya.
- Wajib ada snapshot LANTAI_KOSONG dari Script 05 terbaru agar referensi lantai lokal aktif.
- Jika snapshot LANTAI_KOSONG tidak ditemukan, script akan fallback ke Z_ref scalar dan memberi warning.
- Validasi ROI lantai 05B otomatis menyesuaikan kondisi: lantai terlihat penuh, terlihat sebagian, atau tertutup seluruhnya oleh muatan.

Kontrol:
[S]   Simpan snapshot lengkap jika status VALID
[F]   Reset buffer depth
[M]   Ubah metadata sampel
[ESC] Keluar
"""

import os
import re
import csv
from datetime import datetime

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise RuntimeError(
        "PyTorch belum terpasang atau tidak bisa di-import.\n"
        "Install PyTorch terlebih dahulu, lalu jalankan ulang script ini.\n"
        f"Detail error: {e}"
    )


# =========================================================
# KONFIGURASI PATH
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH_CANDIDATES = [
    os.path.join(BASE_DIR, "hasil_training_unet_colab", "model_best.pth"),
    os.path.join(BASE_DIR, "hasil_training_unet", "model_best.pth"),
    os.path.join(BASE_DIR, "model_best.pth"),
    os.path.join(BASE_DIR, "hasil_training_unet_colab", "model_last.pth"),  # fallback saja
]
MODEL_PATH = MODEL_PATH_CANDIDATES[0]
STEREOMAP_XML = os.path.join(BASE_DIR, "hasil_kalibrasi", "stereomap.xml")
ZREF_XML = os.path.join(BASE_DIR, "hasil_kalibrasi", "zref.xml")
ZREF_NPZ = os.path.join(BASE_DIR, "hasil_kalibrasi", "zref.npz")
SNAPSHOT_DEPTH_DIR = os.path.join(BASE_DIR, "hasil_kalibrasi", "snapshot_depth")

OUTPUT_DIR = os.path.join(BASE_DIR, "hasil_integrasi_unet_depth")
SNAPSHOT_INDEX_CSV = os.path.join(OUTPUT_DIR, "index_snapshot_kalibrasi_berat.csv")

# Kamera stereo
CAMERA_ID_KIRI = 0
CAMERA_ID_KANAN = 1
AUTO_DETECT_CAMERA = True
SCAN_CAMERA_MAX_ID = 10

ALLOW_RESIZE_IF_NEEDED = True

# Karena kalibrasi/capture memakai crop Script 06, frame live harus di-crop dulu
# sebelum cv2.remap agar sesuai dengan stereomap.xml.
TERAPKAN_CROP_REALTIME = True
ROI_LEFT = (0.00, 0.05, 1.00, 0.95)
ROI_RIGHT = (0.00, 0.05, 1.00, 0.95)
CAMERA_RAW_WIDTH = 640
CAMERA_RAW_HEIGHT = 480
FPS = 30

# Setting kamera diselaraskan dengan Script 06 Capture ChArUco.
# Tidak semua webcam mendukung properti ini; jika ditolak driver, OpenCV akan mengabaikan.
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

# Validasi cahaya realtime untuk memastikan depth/height dibaca pada kondisi citra yang layak.
# Nilai dibuat konservatif untuk objek batubara yang memang gelap.
VALIDASI_CAHAYA_AKTIF = True
LIGHT_MEAN_MIN = 45.0
LIGHT_MEAN_MAX = 215.0
LIGHT_STEREO_DIFF_MAX = 18.0
LIGHT_FLICKER_MAX = 6.0
LIGHT_DARK_RATIO_MAX = 45.0
LIGHT_BRIGHT_RATIO_MAX = 15.0
LIGHT_CONTRAST_MIN = 18.0
LIGHT_DARK_PIXEL_VALUE = 25
LIGHT_BRIGHT_PIXEL_VALUE = 245
LIGHT_CENTER_ROI = (0.08, 0.12, 0.92, 0.88)
LIGHT_MIN_CONTENT_RATIO = 0.25

# Z_ref scalar dari Script 5 tetap dimuat sebagai fallback.
# Versi ini SELARAS dengan Script 05B terbaru: jika ada snapshot LANTAI_KOSONG,
# tinggi dihitung memakai referensi lantai lokal per-piksel:
#     h(x,y) = depth_lantai_kosong(x,y) - depth_batubara(x,y)
# Ini mengurangi error akibat kemiringan kamera/perspektif, ketika Z_ref global berbeda
# dari depth lantai pada ROI lokal.
PAKAI_ZREF_VERTICAL_JIKA_ADA = False
PAKAI_REFERENSI_LANTAI_LOKAL = True
FLOOR_REFERENCE_SNAPSHOT_TYPE = "LANTAI_KOSONG"
FLOOR_REFERENCE_FALLBACK_SCALAR = True
FLOOR_REFERENCE_SMOOTH_KSIZE = 9  # ganjil; 0/1 = tidak smoothing

# Depth valid range
# PENTING: rentang ini HARUS mencakup jarak aktual kamera ke permukaan
# gerbong/batubara. Jarak kamera ke lantai gerbong pada setup ini sekitar
# 50-60 cm (lihat proposal: tinggi kamera 60 cm, tilt 10-15 derajat).
# Sebelumnya DEPTH_MAX_M=0.50 (50cm) memotong/menolak piksel pada area
# batubara karena depth aktualnya bisa mendekati atau melebihi 50cm,
# sehingga area itu dianggap "invalid" walau mask U-Net mendeteksinya
# dengan benar -- inilah penyebab valid_percent=0.0% pada dashboard
# sebelumnya, BUKAN karena mask salah posisi.
# Beri margin di kedua sisi: permukaan batubara (lebih dekat ke kamera,
# karena menumpuk dari lantai) dan lantai kosong (paling jauh).
DEPTH_MIN_M = 0.20   # 20 cm -- margin aman jika muatan menumpuk tinggi/overload
DEPTH_MAX_M = 0.70   # 70 cm -- margin aman di atas jarak lantai 60cm
USE_ABS_DEPTH = True

# SGBM parameter
# CATATAN HASIL DIAGNOSIS (lihat debug log): disp_float menunjukkan banyak
# nilai -2048 (kode OpenCV untuk "tidak match") dan valid_depth hanya
# 10-17% dari SELURUH frame -- bukan cuma di area batubara. Ini bukti
# SGBM gagal mencocokkan piksel di hampir seluruh frame, paling parah
# tepat di permukaan batubara yang gelap dan bertekstur acak/rendah
# kontras. Literatur stereo matching menyebutkan masalah ini sebagai
# "low texture problem": piksel pada area rendah-tekstur sulit dicocokkan
# karena tidak ada informasi korespondensi langsung antara citra kiri dan
# kanan (lihat referensi di akhir file). Parameter di bawah disetel ulang
# untuk kasus ini -- bukan nilai default generik.
SGBM_MIN_DISP = 0
SGBM_NUM_DISP = 128   # diturunkan dari 160 -- rentang jarak aktual (20-70cm)
                       # tidak butuh rentang disparitas selebar sebelumnya;
                       # numDisparities besar yang tidak perlu memperbesar
                       # ruang pencarian SGBM dan meningkatkan mismatch acak
SGBM_BLOCK_SIZE = 11   # dinaikkan dari 9 -- block lebih besar mengumpulkan
                       # lebih banyak piksel tetangga per perbandingan,
                       # membantu pencocokan saat tekstur lokal lemah
                       # (tetap di rentang wajar 3-11 sesuai dokumentasi OpenCV)
SGBM_UNIQUENESS_RATIO = 5    # diturunkan dari 10 -- uniquenessRatio tinggi
                              # menolak piksel yang "kurang yakin" cocoknya;
                              # pada permukaan rendah-tekstur, hampir semua
                              # piksel kurang yakin, sehingga nilai tinggi
                              # menolak terlalu banyak piksel valid
SGBM_SPECKLE_WINDOW_SIZE = 150  # dinaikkan -- buang island kecil yang noise
SGBM_SPECKLE_RANGE = 4          # dinaikkan dari 2 -- beri toleransi lebih
                                 # pada variasi disparitas dalam satu region
SGBM_DISP12_MAX_DIFF = 2        # dinaikkan dari 1 -- toleransi konsistensi
                                 # kiri-kanan sedikit lebih longgar

# P1 dan P2 dihitung dari blockSize (lihat buat_sgbm()) -- penalti kehalusan
# yang menentukan SEBERAPA HALUS disparitas dipaksa antar piksel tetangga.
# P1 = penalti perubahan +-1, P2 = penalti perubahan >1 (P2 harus > P1).
# Nilai lebih besar dari default OpenCV dipakai di sini supaya SGBM lebih
# memaksa konsistensi pada area rendah-tekstur, alih-alih membiarkan
# disparitas "lompat-lompat" acak akibat noise.

USE_ABS_DEPTH = True
USE_WLS_IF_AVAILABLE = True
WLS_LAMBDA = 8000.0
WLS_SIGMA = 1.5

# U-Net threshold
UNET_THRESHOLD = 0.5
UNET_IMAGE_WIDTH = 640
UNET_IMAGE_HEIGHT = 480

# Height filter
HEIGHT_MIN_M = 0.0

# Tinggi fisik gerbong miniatur = 7,5 cm. Sistem tetap dibuat bisa membaca overload
# sampai 12 cm, tetapi evaluasi/dashboard tidak lagi menganggap spike tunggal sebagai
# tinggi muatan. Nilai representatif yang dipakai untuk keputusan adalah persentil
# robust (P98), sedangkan max mentah tetap disimpan sebagai informasi debug.
WAGON_INNER_HEIGHT_CM = 7.5
OVERLOAD_HEIGHT_LIMIT_CM = 15.0
HEIGHT_MAX_M = OVERLOAD_HEIGHT_LIMIT_CM / 100.0

# Filter outlier height map. Ini penting saat gerbong hampir penuh: lantai tidak terlihat,
# permukaan batubara dominan, dan SGBM kadang menghasilkan spike lokal. Filter ini
# tidak memotong kemampuan overload karena batas akhirnya tetap 12 cm.
HEIGHT_MEDIAN_FILTER_ACTIVE = True
HEIGHT_MEDIAN_FILTER_KSIZE = 5
HEIGHT_ROBUST_CAP_ACTIVE = True
HEIGHT_ROBUST_CAP_PERCENTILE = 99.5
HEIGHT_ROBUST_CAP_MARGIN_CM = 0.30
HEIGHT_REPRESENTATIVE_PERCENTILE = 98.0

# Jika metadata sampel menunjukkan kondisi normal/penuh tetapi bukan overload, tinggi dibatasi
# sampai tinggi fisik gerbong 7,5 cm agar noise pada muatan hampir penuh tidak dianggap overload.
# Jika metadata berisi kata "overload" atau target berat >= 3200 g, batas tetap 12 cm.
CAP_NORMAL_HEIGHT_BY_METADATA = False
NORMAL_HEIGHT_CAP_MARGIN_CM = 0.00
OVERLOAD_TARGET_WEIGHT_G = 2250.0

# Kriteria validasi realtime agar tampilan mudah dipahami.
# Angka ini bukan standar OpenCV mutlak, tetapi batas operasional sistem miniatur.
WAGON_HEIGHT_LIMIT_CM = OVERLOAD_HEIGHT_LIMIT_CM
MIN_VALID_HEIGHT_PERCENT = 70.0
MIN_FLOOR_REFERENCE_VALID_PERCENT = 85.0
# Pada versi ini, validitas FLOOR MAP LOKAL secara keseluruhan hanya menjadi INFO,
# bukan syarat blokir simpan. Alasannya: floor map dari Script 05 bisa valid rendah
# di seluruh frame, tetapi area validasi ROI lantai 05B dan area batubara realtime
# tetap valid. Syarat utama sekarang: ROI lantai 05B realtime sesuai acuan + height map valid.
FLOOR_REFERENCE_GLOBAL_VALID_IS_FATAL = False
MIN_COAL_PIXELS = 50

# Validasi lantai realtime berbasis hasil Script 05B.
# Tujuan: saat Script 11 berjalan, sistem membaca ROI lantai acuan yang sama
# dengan evaluasi 05B, lalu membandingkan depth realtime dengan nilai acuan
# misalnya aktual 54.50 cm / sistem 54.58 cm. Jika tidak mendekati acuan,
# dashboard memberi peringatan agar posisi kamera/gerbong/cahaya disesuaikan
# sebelum snapshot disimpan.
PAKAI_VALIDASI_LANTAI_05B_REALTIME = True
EVAL05B_DIR = os.path.join(BASE_DIR, "hasil_evaluasi_stereo_05b_mode_ukur")
EVAL05B_TEMPLATE_CSV = os.path.join(EVAL05B_DIR, "data_evaluasi_stereo_05b_mode_ukur.csv")
EVAL05B_HASIL_CSV = os.path.join(EVAL05B_DIR, "hasil_evaluasi_stereo_05b_mode_ukur.csv")
FLOOR_RT_TOLERANSI_CM = 1.0
FLOOR_RT_MIN_VALID_ROI_PERCENT = 70.0
# Jika True, target realtime memakai nilai aktual manual dari 05B (misalnya 54.50 cm).
# Jika False, target memakai nilai sistem hasil 05B (misalnya 54.58 cm).
FLOOR_RT_TARGET_PAKAI_AKTUAL = True

# Jika muatan/batubara menutup ROI lantai acuan 05B, validasi lantai realtime
# TIDAK boleh dipakai sebagai syarat gagal, karena ROI tersebut memang bukan
# lantai lagi. Pada kondisi penuh/500 gram/permukaan rata, lantai sering tertutup
# muatan sehingga depth ROI menjadi lebih dekat dan akan terlihat "tidak sesuai"
# meskipun sistem sebenarnya benar.
FLOOR_RT_LEWATI_JIKA_ROI_TERTUTUP_MASK = True

# Mode otomatis untuk semua kondisi lantai pada ROI 05B:
# 1) LANTAI_TERLIHAT_PENUH     -> ROI tidak tertutup mask; depth ROI lantai harus cocok acuan 05B.
# 2) LANTAI_TERLIHAT_SEBAGIAN  -> sebagian ROI tertutup mask; cek memakai bagian ROI yang masih terlihat.
# 3) LANTAI_TERTUTUP_MUATAN    -> ROI hampir/seluruhnya tertutup; cek lantai realtime dilewati.
# Ini diperlukan karena pada muatan rata/penuh/overload, lantai gerbong memang sering tidak terlihat.
FLOOR_RT_MASK_COVERAGE_PARTIAL_PERCENT = 8.0    # mulai dianggap tertutup sebagian
FLOOR_RT_MASK_COVERAGE_FULL_PERCENT = 60.0       # dianggap tertutup dominan/penuh
FLOOR_RT_MIN_VISIBLE_FLOOR_VALID_PERCENT = 25.0  # minimal piksel lantai terlihat untuk cek lokal
FLOOR_RT_MASK_COVERAGE_SKIP_PERCENT = FLOOR_RT_MASK_COVERAGE_FULL_PERCENT  # kompatibilitas lama

# Stabilisasi depth realtime. Script 05 sudah memakai multi-frame; Script 11 juga perlu
# distabilkan agar height map tidak meloncat karena 1 frame SGBM yang buruk.
USE_TEMPORAL_DEPTH_MEDIAN = True
TEMPORAL_DEPTH_FRAMES = 7
TEMPORAL_MIN_VALID_FRAMES = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Debug console. Default False agar terminal tidak ramai saat pemakaian normal.
DEBUG_DIAGNOSTIK_AWAL = False

# =========================================================
# KONFIGURASI TAMPILAN DASHBOARD
# =========================================================
# Ukuran panel dibuat tetap supaya semua panel terlihat di layar laptop.
DASH_PANEL_W = 380
DASH_PANEL_H = 255
DASH_FOOTER_H = 220
DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 720
DISPLAY_MIRROR_HORIZONTAL = True
WINDOW_NAME = "Integrasi U-Net + Depth + Height VALIDASI [S jika VALID | F reset | M metadata | ESC]"





# =========================================================
# METADATA SAMPEL KALIBRASI BERAT
# =========================================================
def slugify_text(value, default="NA"):
    """Membuat teks aman untuk nama file: tanpa spasi dan karakter aneh."""
    value = str(value).strip().lower()
    if value == "":
        return default
    value = value.replace(" ", "_")
    value = re.sub(r"[^a-z0-9_\-]+", "", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or default


def input_float_optional(prompt, default=None):
    txt = input(prompt).strip()
    if txt == "":
        return default
    try:
        return float(txt.replace(",", "."))
    except Exception:
        print("[WARNING] Input angka tidak valid. Nilai dikosongkan.")
        return default


def input_int_optional(prompt, default=None):
    txt = input(prompt).strip()
    if txt == "":
        return default
    try:
        return int(float(txt.replace(",", ".")))
    except Exception:
        print("[WARNING] Input angka tidak valid. Nilai dikosongkan.")
        return default


def input_metadata_sampel(default=None):
    """
    Metadata ini hanya untuk menata dataset kalibrasi berat.
    Berat aktual tetap diisi nanti pada CSV hasil Script 12/13B agar sesuai timbangan digital.
    """
    if default is None:
        default = {}

    print("\n" + "=" * 78)
    print("METADATA SAMPEL KALIBRASI BERAT")
    print("Isi sesuai kondisi muatan saat ini. Tekan ENTER jika tidak ingin diisi.")
    print("Contoh sample_id: B10_T0500_R01_rata atau B15_T3200_R02_overload")
    print("=" * 78)

    baseline_default = default.get("baseline_cm")
    target_default = default.get("target_weight_g")
    rep_default = default.get("repetition")

    baseline_cm = input_float_optional(
        f"Baseline kamera cm [default {baseline_default if baseline_default is not None else 'kosong'}]: ",
        baseline_default,
    )
    target_weight_g = input_float_optional(
        f"Target berat gram [default {target_default if target_default is not None else 'kosong'}]: ",
        target_default,
    )
    condition = input(f"Kondisi muatan [default {default.get('condition','')}]: ").strip() or default.get("condition", "")
    profile = input(f"Profil muatan [default {default.get('profile','')}]: ").strip() or default.get("profile", "")
    repetition = input_int_optional(
        f"Pengulangan ke- [default {rep_default if rep_default is not None else 'kosong'}]: ",
        rep_default,
    )
    notes = input(f"Catatan [default {default.get('notes','')}]: ").strip() or default.get("notes", "")

    meta = {
        "baseline_cm": baseline_cm,
        "target_weight_g": target_weight_g,
        "condition": condition,
        "profile": profile,
        "repetition": repetition,
        "actual_weight_g": "",  # diisi nanti dari timbangan digital pada CSV Script 12/13B
        "notes": notes,
    }
    print("\n[INFO] Metadata aktif:")
    print(f"       baseline_cm     : {meta['baseline_cm']}")
    print(f"       target_weight_g : {meta['target_weight_g']}")
    print(f"       condition       : {meta['condition']}")
    print(f"       profile         : {meta['profile']}")
    print(f"       repetition      : {meta['repetition']}")
    print("       actual_weight_g : dikosongkan; isi nanti dari timbangan digital")
    print("=" * 78)
    return meta


def buat_sample_id(counter, waktu, metadata=None):
    """Membuat sample_id rapi untuk file Script 11."""
    metadata = metadata or {}
    b = metadata.get("baseline_cm")
    t = metadata.get("target_weight_g")
    r = metadata.get("repetition")
    profile = slugify_text(metadata.get("profile", ""), default="profil")

    if b is not None and t is not None:
        try:
            b_int = int(round(float(b)))
            t_int = int(round(float(t)))
            r_int = int(r) if r is not None else int(counter)
            return f"B{b_int:02d}_T{t_int:04d}_R{r_int:02d}_{profile}_{waktu}"
        except Exception:
            pass
    return f"integrasi_{counter:03d}_{waktu}"


def tulis_index_snapshot(prefix, path_npz, metadata, stats, zref_m):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fields = [
        "timestamp", "sample_id", "npz_path", "baseline_cm", "target_weight_g",
        "actual_weight_g", "condition", "profile", "repetition", "zref_m", "zref_cm",
        "coal_pixels", "valid_height_pixels", "valid_height_percent",
        "mean_height_cm", "median_height_cm", "max_height_cm", "notes"
    ]
    exists = os.path.exists(SNAPSHOT_INDEX_CSV)
    with open(SNAPSHOT_INDEX_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sample_id": prefix,
            "npz_path": path_npz,
            "baseline_cm": "" if metadata.get("baseline_cm") is None else metadata.get("baseline_cm"),
            "target_weight_g": "" if metadata.get("target_weight_g") is None else metadata.get("target_weight_g"),
            "actual_weight_g": metadata.get("actual_weight_g", ""),
            "condition": metadata.get("condition", ""),
            "profile": metadata.get("profile", ""),
            "repetition": "" if metadata.get("repetition") is None else metadata.get("repetition"),
            "zref_m": zref_m,
            "zref_cm": zref_m * 100.0,
            "coal_pixels": stats.get("coal_pixels", 0),
            "valid_height_pixels": stats.get("valid_pixels", 0),
            "valid_height_percent": stats.get("valid_percent", 0.0),
            "mean_height_cm": stats.get("mean_cm", 0.0),
            "median_height_cm": stats.get("median_cm", 0.0),
            "max_height_cm": stats.get("max_cm", 0.0),
            "notes": metadata.get("notes", ""),
        })

# =========================================================
# MODEL U-NET — SAMA DENGAN TRAINING
# =========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# =========================================================
# UTILITAS XML
# =========================================================
def _node_empty(node):
    try:
        return node.empty()
    except Exception:
        return True


def buka_xml_read(path_xml):
    if not os.path.exists(path_xml):
        raise FileNotFoundError(f"File XML tidak ditemukan: {path_xml}")
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


# =========================================================
# LOAD MODEL, STEREOMAP, Z_REF
# =========================================================
def pilih_model_path():
    global MODEL_PATH
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    for p in MODEL_PATH_CANDIDATES:
        if os.path.exists(p):
            MODEL_PATH = p
            return p
    raise FileNotFoundError(
        "Model U-Net tidak ditemukan. Letakkan model_best.pth di salah satu lokasi berikut:\n"
        + "\n".join(MODEL_PATH_CANDIDATES)
    )


def load_model_unet():
    global UNET_THRESHOLD, UNET_IMAGE_WIDTH, UNET_IMAGE_HEIGHT

    model_path = pilih_model_path()
    print(f"[INFO] Load model U-Net: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=DEVICE)

    if "threshold" in checkpoint:
        UNET_THRESHOLD = float(checkpoint["threshold"])
    if "image_width" in checkpoint:
        UNET_IMAGE_WIDTH = int(checkpoint["image_width"])
    if "image_height" in checkpoint:
        UNET_IMAGE_HEIGHT = int(checkpoint["image_height"])

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)

    # Mendukung dua format file:
    # 1) checkpoint dict dari Script 09: {"model_state_dict": ..., "threshold": ...}
    # 2) state_dict mentah jika model disimpan langsung dengan torch.save(model.state_dict()).
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print("[OK] Model U-Net dimuat")
    print(f"     Device    : {DEVICE}")
    print(f"     Threshold : {UNET_THRESHOLD}")
    print(f"     U-Net size: {UNET_IMAGE_WIDTH} x {UNET_IMAGE_HEIGHT}")
    return model


def load_stereomap():
    fs = buka_xml_read(STEREOMAP_XML)
    mapLx = baca_matrix(fs, "stereoMapL_x")
    mapLy = baca_matrix(fs, "stereoMapL_y")
    mapRx = baca_matrix(fs, "stereoMapR_x")
    mapRy = baca_matrix(fs, "stereoMapR_y")
    Q = baca_matrix(fs, "Q")

    image_width = baca_int(fs, "image_width", default=mapLx.shape[1])
    image_height = baca_int(fs, "image_height", default=mapLx.shape[0])
    fs.release()

    print("[OK] Stereomap dimuat")
    print(f"     File       : {STEREOMAP_XML}")
    print(f"     Image size : {(image_width, image_height)}")
    print(f"     Map kiri   : {mapLx.shape} / {mapLy.shape}")
    print(f"     Map kanan  : {mapRx.shape} / {mapRy.shape}")
    print(f"     Q matrix   : {Q.shape}")

    return {
        "mapLx": mapLx,
        "mapLy": mapLy,
        "mapRx": mapRx,
        "mapRy": mapRy,
        "Q": Q,
        "image_size": (image_width, image_height),
    }


def load_zref():
    if os.path.exists(ZREF_XML):
        fs = buka_xml_read(ZREF_XML)
        zref_m = baca_float(fs, "zref_m", default=None)
        fs.release()
        if zref_m is not None:
            print(f"[OK] Z_ref dimuat dari XML: {ZREF_XML}")
            print(f"     Z_ref = {zref_m:.4f} m | {zref_m*100:.2f} cm")
            return float(zref_m)

    if os.path.exists(ZREF_NPZ):
        data = np.load(ZREF_NPZ)
        if PAKAI_ZREF_VERTICAL_JIKA_ADA and "zref_vertical_m" in data:
            zref_arr = np.asarray(data["zref_vertical_m"]).astype(float).ravel()
            zref_m = float(zref_arr[0])
            print("[INFO] Memakai zref_vertical_m dari NPZ")
        elif "zref_m" in data:
            zref_arr = np.asarray(data["zref_m"]).astype(float).ravel()
            zref_m = float(zref_arr[0])
        else:
            zref_m = None

        if zref_m is not None:
            print(f"[OK] Z_ref dimuat dari NPZ: {ZREF_NPZ}")
            print(f"     Z_ref = {zref_m:.4f} m | {zref_m*100:.2f} cm")
            return zref_m

    raise FileNotFoundError(
        "Z_ref belum ditemukan. Jalankan dulu Script 5 untuk mengambil Z_ref gerbong kosong.\n"
        f"Dicari: {ZREF_XML} atau {ZREF_NPZ}"
    )


def _npz_string_value(data, key, default=""):
    if key not in data:
        return default
    try:
        arr = np.asarray(data[key]).ravel()
        if arr.size == 0:
            return default
        return str(arr[0])
    except Exception:
        return default


def _resize_float_map(mat, image_size, interpolation=cv2.INTER_LINEAR):
    target_w, target_h = image_size
    h, w = mat.shape[:2]
    if (w, h) == (target_w, target_h):
        return mat
    return cv2.resize(mat, (target_w, target_h), interpolation=interpolation)


def _smooth_floor_reference(floor_m, floor_valid, fallback_m):
    """Haluskan floor reference agar tidak terlalu noisy, tetapi tetap aman."""
    floor = floor_m.astype(np.float32).copy()
    valid = floor_valid & np.isfinite(floor)
    if FLOOR_REFERENCE_FALLBACK_SCALAR and np.isfinite(fallback_m):
        floor[~valid] = np.float32(fallback_m)
    else:
        floor[~valid] = 0.0

    k = int(FLOOR_REFERENCE_SMOOTH_KSIZE)
    if k >= 3:
        if k % 2 == 0:
            k += 1
        try:
            floor = cv2.medianBlur(floor, k)
        except Exception:
            floor = cv2.GaussianBlur(floor, (k, k), 0)
    return floor.astype(np.float32)


def load_referensi_lantai_lokal(image_size, zref_fallback_m):
    """
    Muat floor reference map dari snapshot LANTAI_KOSONG Script 05.
    Tujuannya sama dengan perbaikan Script 05B: memakai lantai lokal, bukan hanya Z_ref global.
    Return dict atau None.
    """
    if not PAKAI_REFERENSI_LANTAI_LOKAL:
        print("[INFO] Referensi lantai lokal NONAKTIF. Memakai Z_ref scalar saja.")
        return None

    if not os.path.isdir(SNAPSHOT_DEPTH_DIR):
        print(f"[WARNING] Folder snapshot depth tidak ditemukan: {SNAPSHOT_DEPTH_DIR}")
        print("          Script 11 akan fallback ke Z_ref scalar.")
        return None

    kandidat = []
    for nama in sorted(os.listdir(SNAPSHOT_DEPTH_DIR)):
        if not nama.lower().endswith(".npz"):
            continue
        path = os.path.join(SNAPSHOT_DEPTH_DIR, nama)
        try:
            with np.load(path, allow_pickle=True) as data:
                stype = _npz_string_value(data, "snapshot_type", default="")
                if stype == FLOOR_REFERENCE_SNAPSHOT_TYPE or (stype == "" and "001" in nama):
                    if "depth_map" in data:
                        kandidat.append((os.path.getmtime(path), path, stype or "TANPA_METADATA"))
        except Exception as e:
            print(f"[WARNING] Gagal membaca kandidat floor reference {path}: {e}")

    if not kandidat:
        print(f"[WARNING] Tidak ada snapshot {FLOOR_REFERENCE_SNAPSHOT_TYPE} di {SNAPSHOT_DEPTH_DIR}")
        print("          Jalankan Script 05, simpan snapshot LANTAI KOSONG, lalu ulangi Script 11.")
        print("          Untuk sementara Script 11 fallback ke Z_ref scalar.")
        return None

    kandidat.sort(reverse=True)  # ambil yang terbaru
    _, path_ref, stype = kandidat[0]

    with np.load(path_ref, allow_pickle=True) as data:
        depth_map = np.asarray(data["depth_map"], dtype=np.float32)
        if "valid_mask" in data:
            valid = np.asarray(data["valid_mask"]).astype(bool)
        elif "valid_depth" in data:
            valid = np.asarray(data["valid_depth"]).astype(bool)
        else:
            valid = np.isfinite(depth_map)

    depth_map = _resize_float_map(depth_map, image_size, interpolation=cv2.INTER_LINEAR)
    valid = _resize_float_map(valid.astype(np.uint8), image_size, interpolation=cv2.INTER_NEAREST).astype(bool)
    valid &= np.isfinite(depth_map)
    valid &= depth_map >= DEPTH_MIN_M
    valid &= depth_map <= DEPTH_MAX_M

    valid_percent = 100.0 * float(valid.sum()) / float(valid.size) if valid.size else 0.0
    if valid.sum() > 0:
        med_cm = float(np.nanmedian(depth_map[valid]) * 100.0)
    else:
        med_cm = float("nan")

    floor_map = _smooth_floor_reference(depth_map, valid, zref_fallback_m)

    print("[OK] Referensi lantai lokal dimuat dari snapshot Script 05")
    print(f"     File       : {path_ref}")
    print(f"     Tipe       : {stype}")
    print(f"     Valid      : {valid_percent:.1f}%")
    print(f"     Median     : {med_cm:.2f} cm")
    print("     Rumus tinggi: height(x,y) = floor_reference(x,y) - depth_current(x,y)")

    return {
        "path": path_ref,
        "floor_m": floor_map.astype(np.float32),
        "valid": valid.astype(bool),
        "valid_percent": valid_percent,
        "median_cm": med_cm,
    }


# =========================================================
# VALIDASI LANTAI REALTIME BERDASARKAN SCRIPT 05B
# =========================================================
def _to_float_or_none(value):
    try:
        if value is None:
            return None
        s = str(value).strip().replace(",", ".")
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        return float(s)
    except Exception:
        return None


def _to_int_or_none(value):
    try:
        if value is None:
            return None
        s = str(value).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def _baca_csv_dict(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _pilih_baris_lantai_05b():
    """Ambil baris jarak_lantai terbaik dari hasil 05B, fallback ke template CSV."""
    rows = _baca_csv_dict(EVAL05B_HASIL_CSV)
    source = EVAL05B_HASIL_CSV
    if not rows:
        rows = _baca_csv_dict(EVAL05B_TEMPLATE_CSV)
        source = EVAL05B_TEMPLATE_CSV

    kandidat = []
    for r in rows:
        jenis = str(r.get("jenis_uji", "")).lower().strip()
        stype = str(r.get("snapshot_type", "")).upper().strip()
        if jenis != "jarak_lantai" and stype != "LANTAI_KOSONG":
            continue
        x1 = _to_int_or_none(r.get("roi_x1"))
        y1 = _to_int_or_none(r.get("roi_y1"))
        x2 = _to_int_or_none(r.get("roi_x2"))
        y2 = _to_int_or_none(r.get("roi_y2"))
        if None in (x1, y1, x2, y2) or x2 <= x1 or y2 <= y1:
            continue

        aktual = _to_float_or_none(r.get("nilai_aktual_cm"))
        sistem = _to_float_or_none(r.get("nilai_sistem_cm"))
        depth_roi = _to_float_or_none(r.get("depth_roi_median_cm"))
        valid_roi = _to_float_or_none(r.get("valid_roi_persen"))
        status = str(r.get("status", "")).upper()

        target = None
        target_source = ""
        if FLOOR_RT_TARGET_PAKAI_AKTUAL and aktual is not None:
            target = aktual
            target_source = "aktual_05B"
        elif sistem is not None:
            target = sistem
            target_source = "sistem_05B"
        elif depth_roi is not None:
            target = depth_roi
            target_source = "depthROI_05B"
        elif aktual is not None:
            target = aktual
            target_source = "aktual_05B"

        if target is None:
            continue

        score = 0
        if "VALID" in status:
            score += 100
        if valid_roi is not None:
            score += float(valid_roi)
        kandidat.append((score, {
            "csv": source,
            "target_cm": float(target),
            "target_source": target_source,
            "actual_cm": aktual,
            "system_cm": sistem,
            "depth_roi_cm": depth_roi,
            "valid_roi_05b": valid_roi,
            "roi": (int(x1), int(y1), int(x2), int(y2)),
            "status_05b": status,
            "row": r,
        }))

    if not kandidat:
        return None
    kandidat.sort(key=lambda t: t[0], reverse=True)
    return kandidat[0][1]


def load_acuan_lantai_05b(image_size):
    if not PAKAI_VALIDASI_LANTAI_05B_REALTIME:
        print("[INFO] Validasi lantai realtime 05B NONAKTIF.")
        return None
    acuan = _pilih_baris_lantai_05b()
    if acuan is None:
        print("[WARNING] Acuan lantai 05B tidak ditemukan.")
        print(f"          Dicek: {EVAL05B_HASIL_CSV}")
        print(f"          Dicek: {EVAL05B_TEMPLATE_CSV}")
        print("          Jalankan Script 05B sampai jarak_lantai VALID dan ROI lantai tersimpan.")
        return None

    w, h = image_size
    x1, y1, x2, y2 = acuan["roi"]
    x1 = max(0, min(w - 2, x1)); x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 2, y1)); y2 = max(y1 + 1, min(h, y2))
    acuan["roi"] = (x1, y1, x2, y2)

    print("[OK] Acuan lantai realtime dimuat dari Script 05B")
    print(f"     CSV        : {acuan['csv']}")
    print(f"     Target     : {acuan['target_cm']:.2f} cm ({acuan['target_source']})")
    if acuan.get("actual_cm") is not None:
        print(f"     Aktual 05B : {acuan['actual_cm']:.2f} cm")
    if acuan.get("system_cm") is not None:
        print(f"     Sistem 05B : {acuan['system_cm']:.2f} cm")
    print(f"     ROI lantai : x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    print(f"     Toleransi  : ±{FLOOR_RT_TOLERANSI_CM:.2f} cm")
    return acuan


def evaluasi_lantai_realtime(depth_m, valid_depth, acuan_lantai, mask_unet=None):
    """
    Validasi lantai realtime yang otomatis menyesuaikan 3 kondisi:
    1. Lantai terlihat penuh   -> bandingkan depth ROI lantai dengan acuan 05B.
    2. Lantai terlihat sebagian -> bandingkan hanya piksel ROI yang tidak tertutup mask batubara.
    3. Lantai tertutup muatan  -> validasi lantai realtime dilewati; status ditentukan oleh height map.

    Dengan cara ini, Script 11 bisa dipakai untuk kondisi muatan sedikit, sedang,
    penuh rata, maupun overload tanpa dipaksa harus selalu melihat lantai gerbong.
    """
    if acuan_lantai is None:
        return {
            "enabled": False, "ok": False, "reason": "acuan lantai 05B tidak ada",
            "median_cm": np.nan, "target_cm": np.nan, "error_cm": np.nan, "valid_pct": 0.0,
            "visible_valid_pct": 0.0, "mask_coverage_pct": 0.0,
            "check_mode": "TANPA_ACUAN_05B", "roi": None, "color": (0, 220, 255)
        }

    x1, y1, x2, y2 = acuan_lantai["roi"]
    h, w = depth_m.shape[:2]
    x1 = max(0, min(w - 2, x1)); x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 2, y1)); y2 = max(y1 + 1, min(h, y2))

    roi_depth = depth_m[y1:y2, x1:x2]
    roi_valid_all = valid_depth[y1:y2, x1:x2] & np.isfinite(roi_depth)
    total = max(1, int((y2 - y1) * (x2 - x1)))

    roi_mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
    if mask_unet is not None:
        try:
            roi_mask = (mask_unet[y1:y2, x1:x2] == 1)
        except Exception:
            roi_mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)

    mask_coverage_pct = 100.0 * int(roi_mask.sum()) / float(total)
    n_valid_all = int(roi_valid_all.sum())
    valid_pct_all = 100.0 * n_valid_all / float(total)

    # Piksel lantai yang masih terlihat adalah ROI yang tidak tertutup mask batubara.
    visible_floor_valid = roi_valid_all & (~roi_mask)
    n_visible_valid = int(visible_floor_valid.sum())
    visible_valid_pct = 100.0 * n_visible_valid / float(total)

    target_cm = float(acuan_lantai["target_cm"])

    def _median_error(valid_sel):
        if int(valid_sel.sum()) > 0:
            med_cm = float(np.nanmedian(roi_depth[valid_sel]) * 100.0)
        else:
            med_cm = float("nan")
        err_cm = med_cm - target_cm if np.isfinite(med_cm) else float("nan")
        return med_cm, err_cm

    # KONDISI 3 — lantai tertutup dominan/penuh oleh muatan.
    # Tidak masuk akal memaksa ROI lantai sama dengan 05B karena ROI itu sudah menjadi batubara.
    if (FLOOR_RT_LEWATI_JIKA_ROI_TERTUTUP_MASK
            and mask_coverage_pct >= FLOOR_RT_MASK_COVERAGE_FULL_PERCENT):
        median_cm, error_cm = _median_error(roi_valid_all)
        return {
            "enabled": True,
            "ok": True,
            "skipped": True,
            "check_mode": "LANTAI_TERTUTUP_MUATAN",
            "reason": "ROI lantai 05B tertutup muatan; validasi lantai realtime dilewati",
            "median_cm": median_cm,
            "target_cm": target_cm,
            "error_cm": error_cm,
            "valid_pct": valid_pct_all,
            "visible_valid_pct": visible_valid_pct,
            "n_valid": n_valid_all,
            "mask_coverage_pct": mask_coverage_pct,
            "roi": (x1, y1, x2, y2),
            "color": (0, 255, 255),
        }

    # KONDISI 2 — lantai terlihat sebagian. Jika masih ada bagian lantai yang terlihat,
    # validasi memakai piksel non-mask saja supaya tidak tercampur permukaan batubara.
    if mask_coverage_pct >= FLOOR_RT_MASK_COVERAGE_PARTIAL_PERCENT:
        if visible_valid_pct >= FLOOR_RT_MIN_VISIBLE_FLOOR_VALID_PERCENT:
            median_cm, error_cm = _median_error(visible_floor_valid)
            reasons = []
            ok = True
            if not np.isfinite(error_cm):
                ok = False
                reasons.append("depth lantai terlihat tidak terbaca")
            elif abs(error_cm) > FLOOR_RT_TOLERANSI_CM:
                ok = False
                arah = "terlalu jauh" if error_cm > 0 else "terlalu dekat"
                reasons.append(f"lantai terlihat {arah}: error {error_cm:+.2f} cm")

            return {
                "enabled": True,
                "ok": ok,
                "skipped": False,
                "check_mode": "LANTAI_TERLIHAT_SEBAGIAN",
                "reason": "OK - lantai sebagian terlihat" if ok else "; ".join(reasons),
                "median_cm": median_cm,
                "target_cm": target_cm,
                "error_cm": error_cm,
                "valid_pct": valid_pct_all,
                "visible_valid_pct": visible_valid_pct,
                "n_valid": n_visible_valid,
                "mask_coverage_pct": mask_coverage_pct,
                "roi": (x1, y1, x2, y2),
                "color": (0, 255, 0) if ok else (0, 220, 255),
            }

        # Sebagian besar ROI tertutup/terganggu dan bagian lantai terlihat tidak cukup.
        # Agar kondisi muatan penuh/rata tidak terus ditolak, lewati validasi lantai
        # dan andalkan valid height + batas tinggi fisik.
        median_cm, error_cm = _median_error(roi_valid_all)
        return {
            "enabled": True,
            "ok": True,
            "skipped": True,
            "check_mode": "LANTAI_SEBAGIAN_TAPI_TIDAK_CUKUP_TERLIHAT",
            "reason": "lantai hanya terlihat sebagian dan piksel lantai tidak cukup; cek lantai dilewati",
            "median_cm": median_cm,
            "target_cm": target_cm,
            "error_cm": error_cm,
            "valid_pct": valid_pct_all,
            "visible_valid_pct": visible_valid_pct,
            "n_valid": n_valid_all,
            "mask_coverage_pct": mask_coverage_pct,
            "roi": (x1, y1, x2, y2),
            "color": (0, 255, 255),
        }

    # KONDISI 1 — lantai terlihat penuh/tidak tertutup mask, gunakan validasi biasa.
    median_cm, error_cm = _median_error(roi_valid_all)
    reasons = []
    ok = True
    if valid_pct_all < FLOOR_RT_MIN_VALID_ROI_PERCENT:
        ok = False
        reasons.append(f"validROI lantai {valid_pct_all:.1f}% < {FLOOR_RT_MIN_VALID_ROI_PERCENT:.0f}%")
    if not np.isfinite(error_cm):
        ok = False
        reasons.append("depth lantai realtime tidak terbaca")
    elif abs(error_cm) > FLOOR_RT_TOLERANSI_CM:
        ok = False
        arah = "terlalu jauh" if error_cm > 0 else "terlalu dekat"
        reasons.append(f"lantai {arah}: error {error_cm:+.2f} cm")

    return {
        "enabled": True,
        "ok": ok,
        "reason": "OK - lantai terlihat penuh" if ok else "; ".join(reasons),
        "median_cm": median_cm,
        "target_cm": target_cm,
        "error_cm": error_cm,
        "valid_pct": valid_pct_all,
        "visible_valid_pct": visible_valid_pct,
        "n_valid": n_valid_all,
        "mask_coverage_pct": mask_coverage_pct,
        "skipped": False,
        "check_mode": "LANTAI_TERLIHAT_PENUH",
        "roi": (x1, y1, x2, y2),
        "color": (0, 255, 0) if ok else (0, 220, 255),
    }

def gambar_roi_lantai(img, floor_check):
    out = img.copy()
    if not floor_check or floor_check.get("roi") is None:
        return out
    x1, y1, x2, y2 = floor_check["roi"]
    color = floor_check.get("color", (0, 220, 255))
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    label = "ROI LANTAI 05B"
    if floor_check.get("enabled"):
        if floor_check.get("skipped", False):
            label = "ROI 05B TERTUTUP MUATAN"
        else:
            label = f"ROI 05B {floor_check.get('median_cm', float('nan')):.1f}/{floor_check.get('target_cm', float('nan')):.1f}cm"
    cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


# =========================================================
# KAMERA
# =========================================================
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

    print(
        f"[INFO] Setting kamera {nama} diselaraskan dengan Script 06: "
        f"brightness={BRIGHTNESS_VALUE}, contrast={CONTRAST_VALUE}, "
        f"gain={GAIN_VALUE}, WB={WB_TEMPERATURE}"
    )


def buka_kamera_default(camera_id, image_size, backend=None):
    if backend is None:
        cap = cv2.VideoCapture(camera_id)
    else:
        cap = cv2.VideoCapture(camera_id, backend)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(CAMERA_RAW_WIDTH))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(CAMERA_RAW_HEIGHT))
        cap.set(cv2.CAP_PROP_FPS, int(FPS))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        nama_kamera = "KIRI" if int(camera_id) == int(CAMERA_ID_KIRI) else "KANAN"
        samakan_setting_kamera(cap, nama_kamera)
        return cap

    cap.release()
    return None


def buka_dua_kamera(image_size):
    backends = [("DSHOW", cv2.CAP_DSHOW), ("DEFAULT", None), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]

    print("[INFO] Membuka kamera stereo...")

    for nama_backend, backend in backends:
        capL = buka_kamera_default(CAMERA_ID_KIRI, image_size, backend)
        capR = buka_kamera_default(CAMERA_ID_KANAN, image_size, backend)
        if capL is not None and capR is not None:
            print(f"[OK] Kamera terbuka: kiri={CAMERA_ID_KIRI}, kanan={CAMERA_ID_KANAN}, backend={nama_backend}")
            return capL, capR
        if capL is not None:
            capL.release()
        if capR is not None:
            capR.release()

    if not AUTO_DETECT_CAMERA:
        raise RuntimeError("Kamera stereo gagal dibuka dengan ID default.")

    print("[WARNING] ID default gagal. Mencari kamera otomatis...")
    found = []
    for cam_id in range(SCAN_CAMERA_MAX_ID + 1):
        for nama_backend, backend in backends:
            cap = buka_kamera_default(cam_id, image_size, backend)
            if cap is not None:
                print(f"[OK] Kamera ditemukan: ID={cam_id}, backend={nama_backend}")
                found.append((cam_id, cap))
                break
        if len(found) >= 2:
            break

    if len(found) < 2:
        for _, cap in found:
            cap.release()
        raise RuntimeError("Tidak menemukan dua kamera. Cek koneksi kamera kiri dan kanan.")

    print(f"[OK] Auto-camera: kiri={found[0][0]}, kanan={found[1][0]}")
    return found[0][1], found[1][1]


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
    return cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)


def siapkan_frame_input_realtime(frame, image_size, roi, nama):
    if TERAPKAN_CROP_REALTIME:
        return crop_resize_frame(frame, roi, image_size)
    return pastikan_ukuran_frame(frame, image_size, nama)


def pastikan_ukuran_frame(frame, image_size, nama):
    target_w, target_h = image_size
    h, w = frame.shape[:2]
    if (w, h) == image_size:
        return frame

    print(f"[WARNING] Ukuran frame {nama} = {(w, h)}, beda dari stereomap {image_size}")
    if not ALLOW_RESIZE_IF_NEEDED:
        raise RuntimeError("Ukuran frame tidak sesuai stereomap.")
    return cv2.resize(frame, (target_w, target_h))


# =========================================================
# DISPARITY DAN DEPTH
# =========================================================
def buat_sgbm():
    block = SGBM_BLOCK_SIZE
    num_disp = SGBM_NUM_DISP
    channels = 1  # grayL/grayR adalah citra grayscale 1 channel

    # P1/P2 dinaikkan dari rumus dasar OpenCV (8/32 * channels * block^2).
    # Dikali 1.5 supaya SGBM lebih memaksa kehalusan disparitas antar
    # piksel tetangga -- perlu pada permukaan rendah-tekstur seperti
    # batubara, di mana tanpa penalti kehalusan yang cukup besar, SGBM
    # menghasilkan disparitas "lompat-lompat" acak (noise) yang banyak
    # ditolak sebagai tidak valid (-2048).
    P1 = int(8 * channels * block * block * 1.5)
    P2 = int(32 * channels * block * block * 1.5)

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=SGBM_MIN_DISP,
        numDisparities=num_disp,
        blockSize=block,
        P1=P1,
        P2=P2,
        disp12MaxDiff=SGBM_DISP12_MAX_DIFF,
        uniquenessRatio=SGBM_UNIQUENESS_RATIO,
        speckleWindowSize=SGBM_SPECKLE_WINDOW_SIZE,
        speckleRange=SGBM_SPECKLE_RANGE,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    print(f"[INFO] SGBM dibuat -> numDisp={num_disp} blockSize={block} P1={P1} P2={P2}")
    print(f"       uniquenessRatio={SGBM_UNIQUENESS_RATIO} speckleWindow={SGBM_SPECKLE_WINDOW_SIZE} speckleRange={SGBM_SPECKLE_RANGE}")

    use_wls = USE_WLS_IF_AVAILABLE and hasattr(cv2, "ximgproc")
    if use_wls:
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        wls_filter.setLambda(WLS_LAMBDA)
        wls_filter.setSigmaColor(WLS_SIGMA)
        print("[OK] cv2.ximgproc tersedia. WLS filter aktif.")
    else:
        right_matcher = None
        wls_filter = None
        print("[WARNING] WLS tidak aktif. Menggunakan disparity SGBM biasa.")

    return left_matcher, right_matcher, wls_filter, use_wls


def hitung_disparity(grayL, grayR, left_matcher, right_matcher, wls_filter, use_wls):
    dispL = left_matcher.compute(grayL, grayR)

    if use_wls:
        dispR = right_matcher.compute(grayR, grayL)
        filtered = wls_filter.filter(dispL, grayL, None, dispR)
        disp_float = filtered.astype(np.float32) / 16.0
    else:
        disp_float = dispL.astype(np.float32) / 16.0

    return disp_float


def disparity_to_depth(disp_float, Q):
    points_3d = cv2.reprojectImageTo3D(disp_float, Q)
    depth_m = points_3d[:, :, 2].astype(np.float32)
    if USE_ABS_DEPTH:
        depth_m = np.abs(depth_m)
    return depth_m


def buat_valid_depth_mask(disp_float, depth_m):
    # PENTING: OpenCV menandai piksel yang gagal di-match dengan nilai
    # disparitas sangat negatif (umumnya -2048 setelah dibagi 16, sebelum
    # dibagi nilainya -32768). Filter "disp_float > SGBM_MIN_DISP" SAJA
    # tidak cukup ketat untuk membuang ini jika SGBM_MIN_DISP=0, karena
    # -2048 secara teknis memang < 0 jadi seharusnya tertangkap -- tapi
    # untuk jaga-jaga dan transparansi, threshold ditulis eksplisit di sini
    # alih-alih hanya mengandalkan perbandingan terhadap SGBM_MIN_DISP.
    disparity_valid = disp_float > 0.0   # disparitas asli harus positif

    valid = np.isfinite(depth_m)
    valid &= disparity_valid
    valid &= depth_m >= DEPTH_MIN_M
    valid &= depth_m <= DEPTH_MAX_M
    return valid


# =========================================================
# U-NET INFERENCE
# =========================================================
def preprocess_unet(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (UNET_IMAGE_WIDTH, UNET_IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    tensor = torch.from_numpy(img_chw).unsqueeze(0).to(DEVICE, dtype=torch.float32)
    return tensor


@torch.no_grad()
def prediksi_mask_unet(model, img_bgr):
    original_h, original_w = img_bgr.shape[:2]
    tensor = preprocess_unet(img_bgr)
    logits = model(tensor)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask_01 = (prob > UNET_THRESHOLD).astype(np.uint8)
    mask_01 = cv2.resize(mask_01, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return mask_01


# =========================================================
# HEIGHT MAP
# =========================================================
def metadata_mengindikasikan_overload(metadata=None):
    """Deteksi sederhana apakah sampel memang overload berdasarkan metadata operator."""
    metadata = metadata or {}
    teks = " ".join([
        str(metadata.get("condition", "")),
        str(metadata.get("profile", "")),
        str(metadata.get("notes", "")),
    ]).lower()

    if "overload" in teks or "over load" in teks or "heaping" in teks:
        return True

    try:
        target_weight = metadata.get("target_weight_g", None)
        if target_weight is not None and str(target_weight).strip() != "":
            if float(target_weight) >= float(OVERLOAD_TARGET_WEIGHT_G):
                return True
    except Exception:
        pass

    return False


def tinggi_maksimum_operasional_m(metadata=None):
    """
    Batas tinggi operasional:
    - Jika metadata jelas overload -> 12 cm.
    - Jika metadata jelas normal/penuh tapi bukan overload -> 7,5 cm.
    - Jika metadata kosong/tidak jelas -> 12 cm supaya sistem tetap bisa membaca overload.
    """
    if not CAP_NORMAL_HEIGHT_BY_METADATA:
        return HEIGHT_MAX_M

    metadata = metadata or {}
    ada_metadata = any(str(metadata.get(k, "")).strip() != "" for k in ["condition", "profile", "notes", "target_weight_g"])
    if not ada_metadata:
        return HEIGHT_MAX_M

    if metadata_mengindikasikan_overload(metadata):
        return HEIGHT_MAX_M

    return (WAGON_INNER_HEIGHT_CM + NORMAL_HEIGHT_CAP_MARGIN_CM) / 100.0


def hitung_height_map(depth_m, mask_unet, valid_depth, zref_m, floor_reference=None, metadata=None):
    """
    Hitung height map.

    Perbaikan versi ini:
    - Tetap memakai floor reference lokal per-piksel jika tersedia.
    - Tetap bisa membaca overload sampai OVERLOAD_HEIGHT_LIMIT_CM.
    - Mengurangi spike/outlier lokal pada kondisi muatan hampir penuh ketika lantai tidak terlihat.
    - Tidak menghapus piksel tinggi yang melewati batas normal 7,5 cm; batas normal hanya dipakai
      sebagai informasi kategori, sedangkan batas keras sistem adalah batas overload 12 cm.
    """
    coal_area = (mask_unet == 1) & valid_depth

    height_m = np.zeros_like(depth_m, dtype=np.float32)

    if floor_reference is not None and "floor_m" in floor_reference:
        floor_m = floor_reference["floor_m"].astype(np.float32)
        floor_valid = floor_reference.get("valid", np.ones_like(coal_area, dtype=bool)).astype(bool)
        if floor_m.shape != depth_m.shape:
            floor_m = cv2.resize(floor_m, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_LINEAR)
        if floor_valid.shape != depth_m.shape:
            floor_valid = cv2.resize(floor_valid.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        if FLOOR_REFERENCE_FALLBACK_SCALAR and np.isfinite(zref_m):
            floor_eff = floor_m.copy()
            floor_eff[~floor_valid] = np.float32(zref_m)
            area = coal_area
        else:
            floor_eff = floor_m
            area = coal_area & floor_valid

        height_m[area] = floor_eff[area] - depth_m[area]
    else:
        height_m[coal_area] = zref_m - depth_m[coal_area]

    # Bersihkan nilai tidak masuk akal dasar.
    # Nilai negatif berarti permukaan terbaca lebih jauh dari lantai, sehingga bukan tinggi muatan.
    height_m[~np.isfinite(height_m)] = 0.0
    height_m[height_m < HEIGHT_MIN_M] = 0.0

    # Filter median kecil untuk mengurangi spike depth lokal pada batubara gelap/rendah kontras.
    # Filter diterapkan pada peta tinggi, lalu tetap dibatasi pada area batubara saja.
    if HEIGHT_MEDIAN_FILTER_ACTIVE:
        k = int(HEIGHT_MEDIAN_FILTER_KSIZE)
        if k >= 3:
            if k % 2 == 0:
                k += 1
            try:
                height_blur = cv2.medianBlur(height_m.astype(np.float32), k)
                height_m[coal_area] = height_blur[coal_area]
            except Exception:
                pass

    # Batas keras sistem: default overload 12 cm. Jika metadata jelas normal/penuh
    # tetapi bukan overload, batas operasional menjadi 7,5 cm agar muatan hampir penuh
    # tidak naik palsu menjadi 9-10 cm hanya karena spike depth.
    height_cap_m = tinggi_maksimum_operasional_m(metadata)
    height_m[height_m > HEIGHT_MAX_M] = 0.0
    height_m[(height_m > height_cap_m) & coal_area] = height_cap_m

    # Robust cap berbasis persentil hanya untuk membuang lonjakan tunggal yang jauh dari mayoritas.
    # Jika benar-benar overload, banyak piksel akan tinggi, sehingga P99.5 juga ikut tinggi dan tidak dipotong.
    valid_pre = coal_area & (height_m > 0) & np.isfinite(height_m)
    if HEIGHT_ROBUST_CAP_ACTIVE and int(valid_pre.sum()) >= 100:
        vals = height_m[valid_pre]
        cap_m = float(np.percentile(vals, HEIGHT_ROBUST_CAP_PERCENTILE)) + (HEIGHT_ROBUST_CAP_MARGIN_CM / 100.0)
        cap_m = min(cap_m, height_cap_m)
        height_m[(height_m > cap_m) & coal_area] = cap_m

    valid_height = coal_area & (height_m > HEIGHT_MIN_M) & (height_m <= height_cap_m) & np.isfinite(height_m)
    return height_m.astype(np.float32), valid_height

def statistik_height(height_m, valid_height, mask_unet):
    coal_pixels = int((mask_unet == 1).sum())
    valid_pixels = int(valid_height.sum())

    if coal_pixels > 0:
        valid_percent = valid_pixels / coal_pixels * 100.0
    else:
        valid_percent = 0.0

    if valid_pixels == 0:
        return {
            "coal_pixels": coal_pixels,
            "valid_pixels": valid_pixels,
            "valid_percent": valid_percent,
            "mean_cm": 0.0,
            "median_cm": 0.0,
            "p90_cm": 0.0,
            "p95_cm": 0.0,
            "p98_cm": 0.0,
            "representative_max_cm": 0.0,
            "max_cm": 0.0,
            "raw_max_cm": 0.0,
            "height_category": "TIDAK_TERDETEKSI",
        }

    vals_cm = height_m[valid_height] * 100.0
    mean_cm = float(np.mean(vals_cm))
    median_cm = float(np.median(vals_cm))
    p90_cm = float(np.percentile(vals_cm, 90))
    p95_cm = float(np.percentile(vals_cm, 95))
    p98_cm = float(np.percentile(vals_cm, HEIGHT_REPRESENTATIVE_PERCENTILE))
    raw_max_cm = float(np.max(vals_cm))

    # Untuk keputusan, gunakan P98 sebagai tinggi maksimum representatif.
    # Max mentah tetap disimpan untuk melihat apakah ada spike/noise lokal.
    representative_max_cm = p98_cm

    if representative_max_cm <= WAGON_INNER_HEIGHT_CM:
        height_category = "NORMAL_DALAM_GERBONG"
    elif representative_max_cm <= OVERLOAD_HEIGHT_LIMIT_CM:
        height_category = "OVERLOAD_TERUKUR"
    else:
        height_category = "DI_LUAR_BATAS_FISIK"

    return {
        "coal_pixels": coal_pixels,
        "valid_pixels": valid_pixels,
        "valid_percent": valid_percent,
        "mean_cm": mean_cm,
        "median_cm": median_cm,
        "p90_cm": p90_cm,
        "p95_cm": p95_cm,
        "p98_cm": p98_cm,
        "representative_max_cm": representative_max_cm,
        # max_cm dipertahankan sebagai nama lama, tetapi nilainya sekarang tinggi representatif
        # supaya dashboard/evaluasi tidak keliru akibat satu-dua piksel spike.
        "max_cm": representative_max_cm,
        "raw_max_cm": raw_max_cm,
        "height_category": height_category,
    }


def evaluasi_cahaya_realtime(frame_left, frame_right, light_state=None):
    """Menilai apakah cahaya cukup stabil dan seimbang untuk stereo depth."""
    if not VALIDASI_CAHAYA_AKTIF:
        return {"enabled": False, "ok": True, "reason": "validasi cahaya nonaktif"}

    light_state = light_state if light_state is not None else {}

    gray_l = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    def _sample_area(gray):
        h, w = gray.shape[:2]
        rx1, ry1, rx2, ry2 = LIGHT_CENTER_ROI
        x1 = max(0, min(w - 1, int(round(rx1 * w))))
        y1 = max(0, min(h - 1, int(round(ry1 * h))))
        x2 = max(x1 + 1, min(w, int(round(rx2 * w))))
        y2 = max(y1 + 1, min(h, int(round(ry2 * h))))
        crop = gray[y1:y2, x1:x2]
        content = crop[crop > 5]
        min_content = int(crop.size * LIGHT_MIN_CONTENT_RATIO)
        if content.size >= min_content:
            return content
        return crop.reshape(-1)

    area_l = _sample_area(gray_l)
    area_r = _sample_area(gray_r)

    mean_l = float(np.mean(area_l))
    mean_r = float(np.mean(area_r))
    mean_avg = (mean_l + mean_r) / 2.0
    stereo_diff = abs(mean_l - mean_r)

    last_mean_l = light_state.get("last_mean_l")
    last_mean_r = light_state.get("last_mean_r")
    if last_mean_l is None or last_mean_r is None:
        flicker = 0.0
    else:
        flicker = max(abs(mean_l - float(last_mean_l)), abs(mean_r - float(last_mean_r)))
    light_state["last_mean_l"] = mean_l
    light_state["last_mean_r"] = mean_r

    dark_l = 100.0 * float(np.mean(area_l <= LIGHT_DARK_PIXEL_VALUE))
    dark_r = 100.0 * float(np.mean(area_r <= LIGHT_DARK_PIXEL_VALUE))
    bright_l = 100.0 * float(np.mean(area_l >= LIGHT_BRIGHT_PIXEL_VALUE))
    bright_r = 100.0 * float(np.mean(area_r >= LIGHT_BRIGHT_PIXEL_VALUE))
    dark_ratio = max(dark_l, dark_r)
    bright_ratio = max(bright_l, bright_r)

    contrast_l = float(np.std(area_l))
    contrast_r = float(np.std(area_r))
    contrast_min = min(contrast_l, contrast_r)

    reasons = []
    if mean_l < LIGHT_MEAN_MIN or mean_r < LIGHT_MEAN_MIN:
        reasons.append(f"gambar terlalu gelap: mean L={mean_l:.1f}, R={mean_r:.1f}")
    if mean_l > LIGHT_MEAN_MAX or mean_r > LIGHT_MEAN_MAX:
        reasons.append(f"gambar terlalu terang: mean L={mean_l:.1f}, R={mean_r:.1f}")
    if stereo_diff > LIGHT_STEREO_DIFF_MAX:
        reasons.append(f"cahaya kiri-kanan tidak seimbang: diff={stereo_diff:.1f}")
    if flicker > LIGHT_FLICKER_MAX:
        reasons.append(f"cahaya berubah antar-frame: flicker={flicker:.1f}")
    if dark_ratio > LIGHT_DARK_RATIO_MAX:
        reasons.append(f"area terlalu gelap besar: dark={dark_ratio:.1f}%")
    if bright_ratio > LIGHT_BRIGHT_RATIO_MAX:
        reasons.append(f"area terlalu terang besar: bright={bright_ratio:.1f}%")
    if contrast_min < LIGHT_CONTRAST_MIN:
        reasons.append(f"kontras terlalu rendah: std={contrast_min:.1f}")

    ok = len(reasons) == 0
    return {
        "enabled": True,
        "ok": ok,
        "reason": "OK - cahaya stabil dan cukup" if ok else "; ".join(reasons),
        "mean_l": mean_l,
        "mean_r": mean_r,
        "mean_avg": mean_avg,
        "stereo_diff": stereo_diff,
        "flicker": flicker,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "contrast_min": contrast_min,
        "color": (0, 255, 0) if ok else (0, 220, 255),
    }


def stabilisasi_depth_temporal(depth_m, valid_depth, depth_buffer, max_frames=TEMPORAL_DEPTH_FRAMES):
    """Median beberapa frame agar depth lebih stabil untuk Script 11."""
    if not USE_TEMPORAL_DEPTH_MEDIAN:
        return depth_m, valid_depth, {"frames": 1, "ready": True, "mode": "single_frame"}

    depth_for_buffer = depth_m.astype(np.float32).copy()
    depth_for_buffer[~valid_depth] = np.nan
    depth_buffer.append(depth_for_buffer)
    if len(depth_buffer) > int(max_frames):
        depth_buffer.pop(0)

    frames_now = len(depth_buffer)
    if frames_now < max(1, TEMPORAL_MIN_VALID_FRAMES):
        return depth_m, valid_depth, {"frames": frames_now, "ready": False, "mode": "warming_up"}

    stack = np.stack(depth_buffer, axis=0)
    valid_count = np.sum(np.isfinite(stack), axis=0)

    # median ignore NaN; area yang semua NaN tetap NaN lalu dijadikan 0.
    with np.errstate(all="ignore", invalid="ignore"):
        depth_med = np.nanmedian(stack, axis=0).astype(np.float32)

    valid_med = valid_count >= int(TEMPORAL_MIN_VALID_FRAMES)
    valid_med &= np.isfinite(depth_med)
    valid_med &= depth_med >= DEPTH_MIN_M
    valid_med &= depth_med <= DEPTH_MAX_M
    depth_med[~np.isfinite(depth_med)] = 0.0

    return depth_med, valid_med, {
        "frames": frames_now,
        "ready": frames_now >= int(max_frames),
        "mode": f"median_{frames_now}_frame",
    }


def evaluasi_status_height(stats, floor_reference=None, buffer_info=None, floor_check=None, light_check=None):
    """Beri status jelas: VALID atau BELUM VALID, agar user tidak salah menyimpan."""
    reasons = []
    ok = True

    if buffer_info and not buffer_info.get("ready", True):
        ok = False
        reasons.append(f"buffer depth baru {buffer_info.get('frames', 0)} frame")

    if stats.get("coal_pixels", 0) < MIN_COAL_PIXELS:
        ok = False
        reasons.append("mask batubara terlalu kecil / belum terdeteksi")

    if stats.get("valid_percent", 0.0) < MIN_VALID_HEIGHT_PERCENT:
        ok = False
        reasons.append(f"valid height {stats.get('valid_percent', 0.0):.1f}% < {MIN_VALID_HEIGHT_PERCENT:.0f}%")

    if stats.get("max_cm", 0.0) > WAGON_HEIGHT_LIMIT_CM:
        ok = False
        reasons.append(f"tinggi maksimum {stats.get('max_cm', 0.0):.1f} cm > batas {WAGON_HEIGHT_LIMIT_CM:.1f} cm")

    if stats.get("median_cm", 0.0) > WAGON_HEIGHT_LIMIT_CM:
        ok = False
        reasons.append(f"median tinggi {stats.get('median_cm', 0.0):.1f} cm tidak masuk akal")

    floor_valid_percent = None
    floor_median_cm = None
    floor_path = ""
    if floor_reference is not None:
        floor_valid_percent = float(floor_reference.get("valid_percent", 0.0))
        floor_median_cm = float(floor_reference.get("median_cm", 0.0))
        floor_path = str(floor_reference.get("path", ""))
        # Versi SOFT: validitas floor reference global tidak lagi otomatis membuat BELUM VALID.
        # Yang menentukan apakah posisi masih benar adalah ROI lantai realtime 05B.
        if floor_valid_percent < MIN_FLOOR_REFERENCE_VALID_PERCENT and FLOOR_REFERENCE_GLOBAL_VALID_IS_FATAL:
            ok = False
            reasons.append(f"floor reference valid {floor_valid_percent:.1f}% < {MIN_FLOOR_REFERENCE_VALID_PERCENT:.0f}%")
    else:
        # Masih dibuat fatal, karena tanpa floor reference lokal rumus height map berkurang kualitasnya.
        ok = False
        reasons.append("floor reference lokal tidak ditemukan")

    if floor_check is not None and floor_check.get("enabled", False):
        if not floor_check.get("ok", False):
            ok = False
            reasons.append("lantai realtime tidak sesuai 05B: " + str(floor_check.get("reason", "")))
    elif PAKAI_VALIDASI_LANTAI_05B_REALTIME:
        ok = False
        reasons.append("acuan lantai realtime 05B belum dimuat")

    if light_check is not None and light_check.get("enabled", False):
        if not light_check.get("ok", False):
            ok = False
            reasons.append("cahaya belum layak: " + str(light_check.get("reason", "")))
    elif VALIDASI_CAHAYA_AKTIF:
        ok = False
        reasons.append("validasi cahaya belum berjalan")

    status = "VALID - BOLEH SIMPAN" if ok else "BELUM VALID - JANGAN SIMPAN"
    color = (0, 255, 0) if ok else (0, 220, 255)

    return {
        "ok": ok,
        "status": status,
        "reasons": reasons,
        "color": color,
        "floor_valid_percent": floor_valid_percent,
        "floor_median_cm": floor_median_cm,
        "floor_path": floor_path,
    }



# =========================================================
# VISUALISASI
# =========================================================
def colorize_map(mat, valid_mask=None, min_val=None, max_val=None):
    arr = mat.copy().astype(np.float32)

    if valid_mask is None:
        valid_mask = np.isfinite(arr)

    out = np.zeros(arr.shape, dtype=np.uint8)

    if valid_mask.sum() > 0:
        vals = arr[valid_mask]
        if min_val is None:
            min_val = float(np.percentile(vals, 5))
        if max_val is None:
            max_val = float(np.percentile(vals, 95))
        if max_val <= min_val:
            max_val = min_val + 1e-6

        norm = (arr - min_val) / (max_val - min_val)
        norm = np.clip(norm, 0, 1)
        out = (norm * 255).astype(np.uint8)
        out[~valid_mask] = 0

    color = cv2.applyColorMap(out, cv2.COLORMAP_TURBO)
    color[~valid_mask] = (0, 0, 0)
    return color


def overlay_mask(img_bgr, mask_01):
    color = np.zeros_like(img_bgr)
    color[mask_01 == 1] = (0, 255, 0)
    return cv2.addWeighted(img_bgr, 0.65, color, 0.35, 0)


def resize_keep_aspect(img, target_w, target_h):
    """Resize gambar tanpa distorsi, lalu taruh di canvas hitam ukuran tetap."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def buat_panel(img, judul, subjudul=None, panel_w=DASH_PANEL_W, panel_h=DASH_PANEL_H):
    """Membuat panel rapi: header judul + gambar yang selalu muat."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    header_h = 38
    body_h = panel_h - header_h
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    body = resize_keep_aspect(img, panel_w, body_h)
    panel[header_h:panel_h, 0:panel_w] = body

    # Header abu-abu gelap supaya tulisan jelas
    cv2.rectangle(panel, (0, 0), (panel_w, header_h), (35, 35, 35), -1)
    cv2.putText(panel, judul, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    if subjudul:
        cv2.putText(panel, subjudul, (10, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Border tipis
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (80, 80, 80), 1)
    return panel


def tambah_label(img, teks):
    # Fungsi lama tetap disediakan agar pemanggilan lain tidak error.
    return buat_panel(img, teks)


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


def buat_footer(width, stats, zref_m, floor_reference=None, status_info=None, buffer_info=None, floor_check=None, light_check=None):
    footer = np.zeros((DASH_FOOTER_H, width, 3), dtype=np.uint8)
    cv2.rectangle(footer, (0, 0), (width, DASH_FOOTER_H), (20, 20, 20), -1)
    cv2.rectangle(footer, (0, 0), (width, 42), (16, 16, 16), -1)
    cv2.line(footer, (0, 42), (width, 42), (80, 80, 80), 1)

    status_info = status_info or {"ok": False, "status": "BELUM VALID", "reasons": []}
    buffer_info = buffer_info or {"frames": 1, "mode": "single_frame"}

    # Tampilan dibuat sederhana: informasi FloorRef lokal disembunyikan dari dashboard
    # agar tidak membingungkan. FloorRef tetap disimpan di file output sebagai metadata.

    reasons = status_info.get("reasons", [])
    reason_text = "OK - acuan: mode lantai otomatis + valid height" if not reasons else "Cek: " + "; ".join(reasons[:2])

    floor_mode = "ACUAN TIDAK DIMUAT"
    floor_mask = "-"
    floor_status = "-"
    floor_color = (0, 220, 255)
    if floor_check is not None and floor_check.get("enabled", False):
        floor_color = floor_check.get("color", (0, 220, 255))
        floor_mode = str(floor_check.get("check_mode", ""))
        floor_mask = f"{floor_check.get('mask_coverage_pct', 0.0):.1f}%"
        if floor_check.get("skipped", False):
            floor_status = "CEK DILEWATI"
        else:
            floor_status = (
                f"err {floor_check.get('error_cm', float('nan')):+.2f} cm | "
                f"valid {floor_check.get('valid_pct', 0.0):.1f}%"
            )

    light_color = (0, 220, 255)
    light_status = "TIDAK AKTIF"
    light_balance = "-"
    light_quality = "-"
    if light_check is not None and light_check.get("enabled", False):
        light_color = light_check.get("color", (0, 220, 255))
        light_status = "OK" if light_check.get("ok", False) else "CEK"
        light_balance = (
            f"L/R {light_check.get('mean_l', 0.0):.1f}/{light_check.get('mean_r', 0.0):.1f} | "
            f"diff {light_check.get('stereo_diff', 0.0):.1f}"
        )
        light_quality = (
            f"flicker {light_check.get('flicker', 0.0):.1f} | "
            f"dark {light_check.get('dark_ratio', 0.0):.1f}% | "
            f"bright {light_check.get('bright_ratio', 0.0):.1f}%"
        )

    put_text_fit(
        footer,
        f"STATUS: {status_info.get('status', '')}",
        (14, 30),
        0.76,
        status_info.get("color", (0, 220, 255)),
        2,
        width - 28,
    )

    gambar_baris_sel(footer, 48, 38, [
        {"label": "Mean Height", "value": f"{stats['mean_cm']:.2f} cm"},
        {"label": "Median Height", "value": f"{stats['median_cm']:.2f} cm"},
        {"label": "P98 Representatif", "value": f"{stats.get('p98_cm', stats['max_cm']):.2f} cm"},
        {"label": "Raw Max", "value": f"{stats.get('raw_max_cm', stats['max_cm']):.2f} cm"},
        {"label": "Valid Height", "value": f"{stats['valid_percent']:.1f}%"},
    ])

    gambar_baris_sel(footer, 90, 38, [
        {"label": "Mask Batubara", "value": f"{stats['coal_pixels']} px", "color": (230, 230, 230)},
        {"label": "Valid Height Px", "value": f"{stats['valid_pixels']} px", "color": (230, 230, 230)},
        {"label": "Kategori", "value": stats.get("height_category", "NA"), "color": (230, 230, 230)},
        {"label": "Zref Global", "value": f"{zref_m*100:.2f} cm", "color": (230, 230, 230)},
        {"label": "Depth Buffer", "value": f"{buffer_info.get('mode', '')} | {buffer_info.get('frames', 0)} frame", "color": (230, 230, 230)},
    ])

    gambar_baris_sel(footer, 132, 38, [
        {"label": "ROI Lantai 05B", "value": floor_mode, "color": floor_color},
        {"label": "Mask ROI", "value": floor_mask, "color": floor_color},
        {"label": "Cek Lantai", "value": floor_status, "color": floor_color},
        {"label": "Cahaya", "value": light_status, "color": light_color},
        {"label": "Balance Cahaya", "value": light_balance, "color": light_color},
    ])

    gambar_baris_sel(footer, 174, 38, [
        {"label": "Detail Cahaya", "value": light_quality, "color": light_color},
        {"label": "Alasan Status", "value": reason_text, "color": (0, 180, 255) if reasons else (0, 255, 0)},
        {"label": "Kontrol", "value": "S=simpan | F=reset | M=metadata | ESC=keluar", "color": (210, 210, 210)},
    ])
    return footer

def resize_dashboard_to_screen(dashboard):
    h, w = dashboard.shape[:2]
    scale = min(DISPLAY_MAX_WIDTH / w, DISPLAY_MAX_HEIGHT / h, 1.0)
    if scale < 1.0:
        return cv2.resize(dashboard, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return dashboard


def mirror_untuk_tampilan(img):
    return cv2.flip(img, 1) if DISPLAY_MIRROR_HORIZONTAL else img


def buat_dashboard(rectL, rectR, mask_unet, depth_m, valid_depth, height_m, valid_height, stats, zref_m=None, floor_reference=None, status_info=None, buffer_info=None, floor_check=None, light_check=None, mirror_display=False):
    """Dashboard 2x3 yang lebih rapi dan seluruh panel pasti terlihat."""
    mask_vis = cv2.cvtColor((mask_unet * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = overlay_mask(rectL, mask_unet)
    depth_color = colorize_map(depth_m, valid_depth, DEPTH_MIN_M, DEPTH_MAX_M)
    height_color = colorize_map(height_m, valid_height, HEIGHT_MIN_M, HEIGHT_MAX_M)

    rectL_show = gambar_roi_lantai(rectL, floor_check)
    overlay_show = gambar_roi_lantai(overlay, floor_check)
    depth_color_show = gambar_roi_lantai(depth_color, floor_check)

    # Supaya mask tetap terlihat walau area kecil, beri kontur hijau pada mask visual.
    contours, _ = cv2.findContours(mask_unet.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_vis_contour = mask_vis.copy()
    if contours:
        cv2.drawContours(mask_vis_contour, contours, -1, (0, 255, 0), 2)

    if mirror_display:
        rectL_show = mirror_untuk_tampilan(rectL_show)
        rectR = mirror_untuk_tampilan(rectR)
        overlay_show = mirror_untuk_tampilan(overlay_show)
        mask_vis_contour = mirror_untuk_tampilan(mask_vis_contour)
        depth_color_show = mirror_untuk_tampilan(depth_color_show)
        height_color = mirror_untuk_tampilan(height_color)

    p1 = buat_panel(rectL_show, "KIRI RECTIFIED", "kotak = ROI lantai acuan 05B")
    p2 = buat_panel(rectR, "KANAN RECTIFIED")
    p3 = buat_panel(overlay_show, "OVERLAY MASK", "mask hijau = batubara")
    p4 = buat_panel(mask_vis_contour, "MASK U-NET", f"coal pixels: {stats['coal_pixels']}")
    p5 = buat_panel(depth_color_show, "DEPTH MAP", f"range: {DEPTH_MIN_M:.2f}-{DEPTH_MAX_M:.2f} m")
    p6 = buat_panel(
        height_color,
        "HEIGHT MAP",
        f"valid: {stats['valid_percent']:.1f}% | P98 {stats.get('p98_cm', stats['max_cm']):.1f}cm | raw {stats.get('raw_max_cm', stats['max_cm']):.1f}cm"
    )

    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])

    zref_val = 0.0 if zref_m is None else zref_m
    footer = buat_footer(row1.shape[1], stats, zref_val, floor_reference=floor_reference, status_info=status_info, buffer_info=buffer_info, floor_check=floor_check, light_check=light_check)
    dashboard = np.vstack([row1, row2, footer])
    return dashboard, overlay, depth_color, height_color, mask_vis_contour


# =========================================================
# SIMPAN OUTPUT
# =========================================================
def simpan_snapshot(counter, rectL, rectR, mask_unet, depth_m, valid_depth, height_m, valid_height, dashboard, overlay, depth_color, height_color, stats, zref_m, metadata=None, floor_reference=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata = metadata or {}
    waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = buat_sample_id(counter, waktu, metadata)

    path_dashboard = os.path.join(OUTPUT_DIR, f"{prefix}_dashboard.png")
    path_rectL = os.path.join(OUTPUT_DIR, f"{prefix}_rectified_left.png")
    path_rectR = os.path.join(OUTPUT_DIR, f"{prefix}_rectified_right.png")
    path_mask01 = os.path.join(OUTPUT_DIR, f"{prefix}_mask_unet_01.png")
    path_maskvis = os.path.join(OUTPUT_DIR, f"{prefix}_mask_unet_visual.png")
    path_overlay = os.path.join(OUTPUT_DIR, f"{prefix}_overlay.png")
    path_depth = os.path.join(OUTPUT_DIR, f"{prefix}_depth_color.png")
    path_height = os.path.join(OUTPUT_DIR, f"{prefix}_height_color.png")
    path_npz = os.path.join(OUTPUT_DIR, f"{prefix}_data.npz")
    path_txt = os.path.join(OUTPUT_DIR, f"{prefix}_ringkasan.txt")

    cv2.imwrite(path_dashboard, dashboard)
    cv2.imwrite(path_rectL, rectL)
    cv2.imwrite(path_rectR, rectR)
    cv2.imwrite(path_mask01, mask_unet.astype(np.uint8))
    cv2.imwrite(path_maskvis, (mask_unet * 255).astype(np.uint8))
    cv2.imwrite(path_overlay, overlay)
    cv2.imwrite(path_depth, depth_color)
    cv2.imwrite(path_height, height_color)

    floor_ref_mode = "local_floor_map" if floor_reference is not None else "scalar_zref"
    floor_ref_path = "" if floor_reference is None else str(floor_reference.get("path", ""))
    floor_ref_m = np.zeros_like(depth_m, dtype=np.float32)
    floor_ref_valid = np.zeros_like(valid_depth, dtype=np.uint8)
    if floor_reference is not None and "floor_m" in floor_reference:
        floor_ref_m = floor_reference["floor_m"].astype(np.float32)
        floor_ref_valid = floor_reference.get("valid", np.zeros_like(valid_depth, dtype=bool)).astype(np.uint8)
        if floor_ref_m.shape != depth_m.shape:
            floor_ref_m = cv2.resize(floor_ref_m, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_LINEAR)
        if floor_ref_valid.shape != depth_m.shape:
            floor_ref_valid = cv2.resize(floor_ref_valid, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)

    np.savez_compressed(
        path_npz,
        depth_m=depth_m,
        valid_depth=valid_depth.astype(np.uint8),
        mask_unet=mask_unet.astype(np.uint8),
        height_m=height_m,
        valid_height=valid_height.astype(np.uint8),
        zref_m=np.array([zref_m], dtype=np.float32),
        floor_reference_mode=np.array([floor_ref_mode]),
        floor_reference_path=np.array([floor_ref_path]),
        floor_reference_m=floor_ref_m.astype(np.float32),
        floor_reference_valid=floor_ref_valid.astype(np.uint8),
        mean_height_cm=np.array([stats["mean_cm"]], dtype=np.float32),
        median_height_cm=np.array([stats["median_cm"]], dtype=np.float32),
        p90_height_cm=np.array([stats.get("p90_cm", stats["max_cm"])], dtype=np.float32),
        p95_height_cm=np.array([stats.get("p95_cm", stats["max_cm"])], dtype=np.float32),
        p98_height_cm=np.array([stats.get("p98_cm", stats["max_cm"])], dtype=np.float32),
        representative_max_height_cm=np.array([stats.get("representative_max_cm", stats["max_cm"])], dtype=np.float32),
        raw_max_height_cm=np.array([stats.get("raw_max_cm", stats["max_cm"])], dtype=np.float32),
        max_height_cm=np.array([stats["max_cm"]], dtype=np.float32),
        height_category=np.array([stats.get("height_category", "")]),
        sample_id=np.array([prefix]),
        baseline_cm=np.array([np.nan if metadata.get("baseline_cm") is None else metadata.get("baseline_cm")], dtype=np.float32),
        target_weight_g=np.array([np.nan if metadata.get("target_weight_g") is None else metadata.get("target_weight_g")], dtype=np.float32),
        actual_weight_g=np.array([np.nan], dtype=np.float32),
        condition=np.array([metadata.get("condition", "")]),
        profile=np.array([metadata.get("profile", "")]),
        repetition=np.array([-1 if metadata.get("repetition") is None else metadata.get("repetition")], dtype=np.int32),
        notes=np.array([metadata.get("notes", "")]),
    )

    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("RINGKASAN INTEGRASI U-NET + DEPTH + REFERENSI LANTAI LOKAL\n")
        f.write("=========================================\n")
        f.write(f"Sample ID        : {prefix}\n")
        f.write(f"Baseline cm      : {metadata.get('baseline_cm', '')}\n")
        f.write(f"Target weight g  : {metadata.get('target_weight_g', '')}\n")
        f.write(f"Actual weight g  : diisi nanti dari timbangan digital\n")
        f.write(f"Condition        : {metadata.get('condition', '')}\n")
        f.write(f"Profile          : {metadata.get('profile', '')}\n")
        f.write(f"Repetition       : {metadata.get('repetition', '')}\n")
        f.write(f"Notes            : {metadata.get('notes', '')}\n")
        f.write(f"Z_ref_m          : {zref_m:.6f}\n")
        f.write(f"Z_ref_cm         : {zref_m*100:.3f}\n")
        f.write(f"Coal pixels      : {stats['coal_pixels']}\n")
        f.write(f"Valid pixels     : {stats['valid_pixels']}\n")
        f.write(f"Valid percent    : {stats['valid_percent']:.3f}%\n")
        f.write(f"Mean height cm   : {stats['mean_cm']:.6f}\n")
        f.write(f"Median height cm : {stats['median_cm']:.6f}\n")
        f.write(f"P95 height cm    : {stats.get('p95_cm', stats['max_cm']):.6f}\n")
        f.write(f"P98 height cm    : {stats.get('p98_cm', stats['max_cm']):.6f}\n")
        f.write(f"Max representatif: {stats.get('representative_max_cm', stats['max_cm']):.6f}\n")
        f.write(f"Raw max height cm: {stats.get('raw_max_cm', stats['max_cm']):.6f}\n")
        f.write(f"Kategori tinggi  : {stats.get('height_category', '')}\n")
        f.write("\nCatatan: script ini menghasilkan height map. Volume final dihitung pada tahap berikutnya.\n")

    tulis_index_snapshot(prefix, path_npz, metadata, stats, zref_m)

    print(f"[OK] Snapshot tersimpan: {prefix}")
    print(f"     Dashboard : {path_dashboard}")
    print(f"     Data NPZ  : {path_npz}")


# =========================================================
# MAIN REALTIME
# =========================================================
def main():
    print("\n" + "=" * 78)
    print("SCRIPT 11 — U-NET + DEPTH + HEIGHT MAP + VALIDASI REALTIME 05B (FLOORREF SOFT)")
    print("=" * 78)
    print(f"Model       : {MODEL_PATH}")
    print(f"Stereomap   : {STEREOMAP_XML}")
    print(f"Z_ref XML   : {ZREF_XML}")
    print(f"Z_ref NPZ   : {ZREF_NPZ}")
    print(f"Snapshot 05 : {SNAPSHOT_DEPTH_DIR}")
    print(f"Output      : {OUTPUT_DIR}")
    print(f"Device      : {DEVICE}")
    print(f"Crop RT     : {TERAPKAN_CROP_REALTIME}")
    print(f"ROI left    : {ROI_LEFT}")
    print(f"ROI right   : {ROI_RIGHT}")
    print("=" * 78)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_unet = load_model_unet()
    stereomap = load_stereomap()
    zref_m = load_zref()
    floor_reference = load_referensi_lantai_lokal(stereomap["image_size"], zref_m)
    acuan_lantai_05b = load_acuan_lantai_05b(stereomap["image_size"])

    # Metadata diisi SEBELUM kamera dibuka agar kamera tidak idle/timeout saat menunggu input.
    metadata_sampel = input_metadata_sampel()

    capL, capR = buka_dua_kamera(stereomap["image_size"])
    left_matcher, right_matcher, wls_filter, use_wls = buat_sgbm()

    mapLx = stereomap["mapLx"]
    mapLy = stereomap["mapLy"]
    mapRx = stereomap["mapRx"]
    mapRy = stereomap["mapRy"]
    Q = stereomap["Q"]
    image_size = stereomap["image_size"]

    print("\n[INFO] Kontrol:")
    print("       S   = simpan snapshot HANYA jika status VALID")
    print("       F   = reset buffer depth temporal setelah menggeser kamera/gerbong")
    print("       M   = ubah metadata sampel")
    print("       ESC = keluar")
    print("       Catatan: jika ROI lantai 05B masih terlihat sebagai lantai, nilainya harus mendekati target 05B.")
    print("       Jika ROI lantai 05B tertutup muatan, validasi lantai dilewati otomatis dan status fokus ke height map.")
    print("       Dashboard disederhanakan; simpan ditentukan oleh ROI05B/tertutup muatan + valid height.")

    counter = 0
    depth_buffer = []
    light_state = {}

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT)

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()

        if not retL or not retR:
            print("[ERROR] Frame kamera gagal dibaca.")
            break

        frameL = siapkan_frame_input_realtime(frameL, image_size, ROI_LEFT, "kiri")
        frameR = siapkan_frame_input_realtime(frameR, image_size, ROI_RIGHT, "kanan")

        rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)
        light_check = evaluasi_cahaya_realtime(rectL, rectR, light_state)

        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

        disp_float = hitung_disparity(grayL, grayR, left_matcher, right_matcher, wls_filter, use_wls)
        depth_m_raw = disparity_to_depth(disp_float, Q)
        valid_depth_raw = buat_valid_depth_mask(disp_float, depth_m_raw)
        depth_m, valid_depth, buffer_info = stabilisasi_depth_temporal(depth_m_raw, valid_depth_raw, depth_buffer)

        mask_unet = prediksi_mask_unet(model_unet, rectL)
        floor_check = evaluasi_lantai_realtime(depth_m, valid_depth, acuan_lantai_05b, mask_unet=mask_unet)
        height_m, valid_height = hitung_height_map(
            depth_m, mask_unet, valid_depth, zref_m,
            floor_reference=floor_reference, metadata=metadata_sampel
        )
        stats = statistik_height(height_m, valid_height, mask_unet)
        status_info = evaluasi_status_height(
            stats, floor_reference=floor_reference, buffer_info=buffer_info,
            floor_check=floor_check, light_check=light_check
        )

        # ===== DEBUG DIAGNOSTIK OPSIONAL =====
        if DEBUG_DIAGNOSTIK_AWAL and counter == 0:
            mask_pixels = int((mask_unet == 1).sum())
            valid_depth_pixels = int(valid_depth.sum())
            overlap_pixels = int(((mask_unet == 1) & valid_depth).sum())

            # Distribusi disparity TANPA nilai ekstrem -2048 (biar tidak
            # mendominasi statistik min/max seperti sebelumnya)
            disp_masuk_akal = disp_float[disp_float > -1000]
            if disp_masuk_akal.size > 0:
                pct = np.percentile(disp_masuk_akal, [5, 25, 50, 75, 95])
                print(f"\n[DEBUG] disp (tanpa -2048) -> count={disp_masuk_akal.size} "
                      f"p5={pct[0]:.1f} p25={pct[1]:.1f} p50={pct[2]:.1f} p75={pct[3]:.1f} p95={pct[4]:.1f}")

            # Berapa persen piksel SELURUH FRAME yang -2048 (gagal total)?
            gagal_total = int((disp_float <= -1000).sum())
            print(f"[DEBUG] piksel gagal match (~-2048) = {gagal_total} dari {disp_float.size} "
                  f"({100*gagal_total/disp_float.size:.1f}%)")

            # Cek KHUSUS di dalam area mask saja -- apa disparity-nya di sana?
            disp_di_mask = disp_float[mask_unet == 1]
            gagal_di_mask = int((disp_di_mask <= -1000).sum())
            print(f"[DEBUG] DI DALAM AREA MASK -> {gagal_di_mask} dari {disp_di_mask.size} piksel gagal match "
                  f"({100*gagal_di_mask/max(disp_di_mask.size,1):.1f}%)")
            if (disp_di_mask > -1000).any():
                disp_mask_valid = disp_di_mask[disp_di_mask > -1000]
                print(f"[DEBUG] DI DALAM AREA MASK, disparity yang berhasil match -> "
                      f"min={disp_mask_valid.min():.1f} max={disp_mask_valid.max():.1f} mean={disp_mask_valid.mean():.1f}")

            # Cek bounding box posisi mask vs posisi area valid_depth
            ys_mask, xs_mask = np.where(mask_unet == 1)
            ys_valid, xs_valid = np.where(valid_depth)
            if len(ys_mask) > 0:
                print(f"[DEBUG] Bounding box MASK       -> x:[{xs_mask.min()}-{xs_mask.max()}] y:[{ys_mask.min()}-{ys_mask.max()}]")
            if len(ys_valid) > 0:
                print(f"[DEBUG] Bounding box VALID_DEPTH-> x:[{xs_valid.min()}-{xs_valid.max()}] y:[{ys_valid.min()}-{ys_valid.max()}]")
            print(f"[DEBUG] Frame size -> width={disp_float.shape[1]} height={disp_float.shape[0]}\n")
        # ===== AKHIR DEBUG DIAGNOSTIK OPSIONAL =====

        dashboard, overlay, depth_color, height_color, mask_vis = buat_dashboard(
            rectL, rectR, mask_unet, depth_m, valid_depth, height_m, valid_height,
            stats, zref_m, floor_reference=floor_reference, status_info=status_info,
            buffer_info=buffer_info, floor_check=floor_check, light_check=light_check
        )

        dashboard_display, _, _, _, _ = buat_dashboard(
            rectL, rectR, mask_unet, depth_m, valid_depth, height_m, valid_height,
            stats, zref_m, floor_reference=floor_reference, status_info=status_info,
            buffer_info=buffer_info, floor_check=floor_check, light_check=light_check,
            mirror_display=True
        )
        dashboard_show = resize_dashboard_to_screen(dashboard_display)
        cv2.imshow(WINDOW_NAME, dashboard_show)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("m") or key == ord("M"):
            metadata_sampel = input_metadata_sampel(metadata_sampel)
        elif key == ord("f") or key == ord("F"):
            depth_buffer.clear()
            print("[INFO] Buffer depth temporal di-reset. Tunggu beberapa frame sampai status kembali valid.")
        elif key == ord("s") or key == ord("S"):
            if not status_info.get("ok", False):
                print("\n[WARNING] Snapshot TIDAK disimpan karena status belum valid.")
                print("          Alasan:")
                for alasan in status_info.get("reasons", []):
                    print(f"          - {alasan}")
                print("          Perbaiki cahaya/floor reference/depth/posisi ROI lantai, tekan F, lalu tunggu VALID.\n")
            else:
                counter += 1
                simpan_snapshot(
                    counter, rectL, rectR, mask_unet, depth_m, valid_depth,
                    height_m, valid_height, dashboard, overlay, depth_color,
                    height_color, stats, zref_m, metadata_sampel, floor_reference=floor_reference
                )

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

    print("\n[SELESAI] Program integrasi selesai.")
    print(f"Output tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


# ==========================================================================
# REFERENSI PERBAIKAN SGBM (gaya APA)
# ==========================================================================
#
# Bradski, G., & Kaehler, A. (melalui dokumentasi OpenCV). StereoSGBM
#     class reference. https://docs.opencv.org/3.4/javadoc/org/opencv/
#     calib3d/StereoSGBM.html
#
# OpenCV Q&A Forum. (n.d.). Stereo SGBM algorithm -- parameter tuning for
#     low contrast surfaces. https://answers.opencv.org/question/213227/
#     stereo-sgbm-algorithm/
#
# Catatan diagnosis (lihat juga histori debug log): permukaan batubara
# tergolong low-texture region pada citra stereo -- area dengan variasi
# tekstur rendah dan kontras rendah membuat algoritma pencocokan stereo
# kekurangan informasi korespondensi langsung antara citra kiri dan kanan,
# sehingga rawan menghasilkan disparitas tidak valid pada area tersebut.
# ==========================================================================
