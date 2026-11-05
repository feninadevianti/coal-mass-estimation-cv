

Output:
hasil_inference_unet/
    images/
    masks_01/
    masks_visual/
    overlays/
    realtime/
    inference_log.csv

Mask:
0 = background
1 = batubara

import os
import csv
from glob import glob
from datetime import datetime

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH_CANDIDATES = [
    os.path.join(BASE_DIR, "hasil_training_unet_colab", "model_best.pth"),
    os.path.join(BASE_DIR, "hasil_training_unet", "model_best.pth"),
]
MODEL_PATH = MODEL_PATH_CANDIDATES[0]

STEREOMAP_XML = os.path.join(BASE_DIR, "hasil_kalibrasi", "stereomap.xml")

MODE = "folder"  # "folder" atau "camera"

INPUT_IMAGE_DIR = os.path.join(BASE_DIR, "dataset_unet_binary_roboflow_FIXED_01", "test", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "hasil_inference_unet")

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
THRESHOLD = 0.5
BASE_CHANNELS = 32

CAMERA_ID_KIRI = 0
CAMERA_RAW_WIDTH = 640
CAMERA_RAW_HEIGHT = 480
FPS = 30

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

# Karena kalibrasi/capture memakai crop Script 06, live camera harus dicrop dulu sebelum remap.
TERAPKAN_CROP_REALTIME = True
ROI_LEFT = (0.00, 0.05, 1.00, 0.95)
ALLOW_RESIZE_IF_NEEDED = True

# Tampilan realtime
DISPLAY_MAX_WIDTH = 1400
DISPLAY_MAX_HEIGHT = 760
DISPLAY_MIRROR_HORIZONTAL = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# MODEL U-NET — HARUS SAMA DENGAN TRAINING
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
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        b = int(base_channels)
        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 16)
        self.up1 = Up(b * 16, b * 8)
        self.up2 = Up(b * 8, b * 4)
        self.up3 = Up(b * 4, b * 2)
        self.up4 = Up(b * 2, b)
        self.outc = nn.Conv2d(b, out_channels, 1)

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
# LOAD MODEL
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
        "Model terbaik tidak ditemukan. Letakkan model_best.pth di salah satu folder berikut:\n"
        + "\n".join(MODEL_PATH_CANDIDATES)
    )


def load_model():
    global IMAGE_WIDTH, IMAGE_HEIGHT, THRESHOLD, BASE_CHANNELS

    model_path = pilih_model_path()
    print(f"[INFO] Load model: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)

    IMAGE_WIDTH = int(checkpoint.get("image_width", IMAGE_WIDTH))
    IMAGE_HEIGHT = int(checkpoint.get("image_height", IMAGE_HEIGHT))
    THRESHOLD = float(checkpoint.get("threshold", THRESHOLD))
    BASE_CHANNELS = int(checkpoint.get("base_channels", BASE_CHANNELS))

    model = UNet(in_channels=3, out_channels=1, base_channels=BASE_CHANNELS).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("[OK] Model berhasil dimuat")
    print(f"     Device       : {DEVICE}")
    print(f"     Image size   : {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    print(f"     Threshold    : {THRESHOLD}")
    print(f"     Base channels: {BASE_CHANNELS}")
    return model


# =========================================================
# PREDIKSI
# =========================================================
def preprocess_image_bgr(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img_norm = img_resize.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    tensor = torch.from_numpy(img_chw).unsqueeze(0).to(DEVICE, dtype=torch.float32)
    return tensor


@torch.no_grad()
def prediksi_mask(model, img_bgr):
    original_h, original_w = img_bgr.shape[:2]
    tensor = preprocess_image_bgr(img_bgr)
    logits = model(tensor)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask_01 = (prob > THRESHOLD).astype(np.uint8)
    mask_01 = cv2.resize(mask_01, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return mask_01, prob


def buat_overlay(img_bgr, mask_01):
    warna = np.zeros_like(img_bgr)
    warna[mask_01 == 1] = (0, 255, 0)
    return cv2.addWeighted(img_bgr, 0.65, warna, 0.35, 0)


def tambah_label(img, teks):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(out, teks, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return out


def buat_folder_output():
    dirs = {
        "images": os.path.join(OUTPUT_DIR, "images"),
        "masks_01": os.path.join(OUTPUT_DIR, "masks_01"),
        "masks_visual": os.path.join(OUTPUT_DIR, "masks_visual"),
        "overlays": os.path.join(OUTPUT_DIR, "overlays"),
        "realtime": os.path.join(OUTPUT_DIR, "realtime"),
    }
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    return dirs


def tulis_log(base_name, image_path, mask_path, overlay_path, coal_pixels, bg_pixels):
    log_path = os.path.join(OUTPUT_DIR, "inference_log.csv")
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "base_name", "image_path", "mask_path", "overlay_path",
                "coal_pixels", "background_pixels", "coal_area_percent", "threshold", "image_size"
            ])
        total = coal_pixels + bg_pixels
        coal_percent = 0.0 if total == 0 else 100.0 * coal_pixels / total
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            base_name, image_path, mask_path, overlay_path,
            coal_pixels, bg_pixels, f"{coal_percent:.6f}", THRESHOLD, f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}"
        ])


def simpan_hasil_inference(img_bgr, mask_01, overlay, base_name, dirs):
    image_path = os.path.join(dirs["images"], f"{base_name}_image.png")
    mask01_path = os.path.join(dirs["masks_01"], f"{base_name}_mask01.png")
    maskvis_path = os.path.join(dirs["masks_visual"], f"{base_name}_mask_visual.png")
    overlay_path = os.path.join(dirs["overlays"], f"{base_name}_overlay.png")

    cv2.imwrite(image_path, img_bgr)
    cv2.imwrite(mask01_path, mask_01.astype(np.uint8))
    cv2.imwrite(maskvis_path, (mask_01 * 255).astype(np.uint8))
    cv2.imwrite(overlay_path, overlay)

    coal_pixels = int((mask_01 == 1).sum())
    bg_pixels = int((mask_01 == 0).sum())
    tulis_log(base_name, image_path, mask01_path, overlay_path, coal_pixels, bg_pixels)

    print(f"[OK] Output tersimpan: {base_name} | coal_pixels={coal_pixels}")


# =========================================================
# MODE FOLDER
# =========================================================
def inference_folder(model):
    if not os.path.isdir(INPUT_IMAGE_DIR):
        raise FileNotFoundError(f"Folder input gambar tidak ditemukan: {INPUT_IMAGE_DIR}")

    dirs = buat_folder_output()
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_paths.extend(glob(os.path.join(INPUT_IMAGE_DIR, ext)))
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise RuntimeError(f"Tidak ada gambar di folder: {INPUT_IMAGE_DIR}")

    print(f"[INFO] Mode folder")
    print(f"       Input : {INPUT_IMAGE_DIR}")
    print(f"       Total : {len(image_paths)} gambar")

    for idx, path_img in enumerate(image_paths, start=1):
        img = cv2.imread(path_img, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARNING] Gagal membaca gambar: {path_img}")
            continue

        mask_01, _ = prediksi_mask(model, img)
        overlay = buat_overlay(img, mask_01)
        name = os.path.splitext(os.path.basename(path_img))[0]
        base_name = f"{idx:04d}_{name}"
        simpan_hasil_inference(img, mask_01, overlay, base_name, dirs)

    print(f"\n[SELESAI] Inference folder selesai. Output: {OUTPUT_DIR}")


# =========================================================
# MODE CAMERA + STEREOMAP
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
            raise RuntimeError(f"Node '{nama_node}' tidak ditemukan di XML.")
        return None
    mat = node.mat()
    if mat is None and wajib:
        raise RuntimeError(f"Node '{nama_node}' tidak bisa dibaca sebagai matrix.")
    return mat


def baca_int(fs, nama_node, default=None):
    node = fs.getNode(nama_node)
    if _node_empty(node):
        return default
    try:
        return int(round(float(node.real())))
    except Exception:
        return default


def baca_float(fs, nama_node, default=None):
    node = fs.getNode(nama_node)
    if _node_empty(node):
        return default
    try:
        return float(node.real())
    except Exception:
        return default


def muat_stereomap_kiri():
    if not os.path.exists(STEREOMAP_XML):
        raise FileNotFoundError(f"stereomap.xml tidak ditemukan: {STEREOMAP_XML}")

    fs = cv2.FileStorage(STEREOMAP_XML, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Gagal membuka XML: {STEREOMAP_XML}")

    mapLx = baca_matrix(fs, "stereoMapL_x")
    mapLy = baca_matrix(fs, "stereoMapL_y")
    image_width = baca_int(fs, "image_width", default=mapLx.shape[1])
    image_height = baca_int(fs, "image_height", default=mapLx.shape[0])
    rms_stereo = baca_float(fs, "rms_stereo", default=None)
    baseline_mm = baca_float(fs, "baseline_mm", default=None)
    fs.release()

    print("[OK] Stereomap kiri dimuat")
    print(f"     Map size   : {(image_width, image_height)}")
    if rms_stereo is not None:
        print(f"     RMS stereo : {rms_stereo:.6f} px")
    if baseline_mm is not None:
        print(f"     Baseline   : {baseline_mm:.3f} mm | {baseline_mm/10.0:.3f} cm")

    return {"mapLx": mapLx, "mapLy": mapLy, "image_size": (image_width, image_height)}


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


def pastikan_ukuran_frame(frame, image_size):
    target_w, target_h = image_size
    h, w = frame.shape[:2]
    if (w, h) == image_size:
        return frame
    if not ALLOW_RESIZE_IF_NEEDED:
        raise RuntimeError(f"Ukuran frame {(w, h)} tidak sesuai stereomap {image_size}")
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def siapkan_frame_input_realtime(frame_raw, image_size):
    if TERAPKAN_CROP_REALTIME:
        return crop_resize_frame(frame_raw, ROI_LEFT, image_size)
    return pastikan_ukuran_frame(frame_raw, image_size)


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


def buka_kamera(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera ID {camera_id} tidak bisa dibuka.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(CAMERA_RAW_WIDTH))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(CAMERA_RAW_HEIGHT))
    cap.set(cv2.CAP_PROP_FPS, int(FPS))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    samakan_setting_kamera(cap, "KIRI")
    return cap


def resize_agar_muat_layar(img):
    h, w = img.shape[:2]
    scale = min(DISPLAY_MAX_WIDTH / w, DISPLAY_MAX_HEIGHT / h, 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def mirror_untuk_tampilan(img):
    return cv2.flip(img, 1) if DISPLAY_MIRROR_HORIZONTAL else img


def inference_camera(model):
    dirs = buat_folder_output()
    stereomap = muat_stereomap_kiri()
    mapLx = stereomap["mapLx"]
    mapLy = stereomap["mapLy"]
    image_size = stereomap["image_size"]
    cap = buka_kamera(CAMERA_ID_KIRI)

    print("\n[INFO] Mode camera realtime")
    print(f"       Crop realtime : {TERAPKAN_CROP_REALTIME}")
    print(f"       ROI kiri      : {ROI_LEFT}")
    print("       S   = simpan frame + mask + overlay")
    print("       ESC = keluar")

    count = 0
    while True:
        ret, frame_raw = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame kamera.")
            break

        frame_input = siapkan_frame_input_realtime(frame_raw, image_size)
        rectified = cv2.remap(frame_input, mapLx, mapLy, cv2.INTER_LINEAR)
        mask_01, _ = prediksi_mask(model, rectified)
        overlay = buat_overlay(rectified, mask_01)
        mask_vis = cv2.cvtColor((mask_01 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        tampil1 = tambah_label(mirror_untuk_tampilan(rectified), "KIRI RECTIFIED")
        tampil2 = tambah_label(mirror_untuk_tampilan(mask_vis), "MASK U-NET 0/1")
        tampil3 = tambah_label(mirror_untuk_tampilan(overlay), "OVERLAY BATUBARA")
        gabung = np.hstack([tampil1, tampil2, tampil3])
        gabung_tampil = resize_agar_muat_layar(gabung)
        cv2.imshow("Inference U-Net Realtime [S=Simpan | ESC=Keluar]", gabung_tampil)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("s") or key == ord("S"):
            count += 1
            waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"realtime_{count:03d}_{waktu}"
            simpan_hasil_inference(rectified, mask_01, overlay, base_name, dirs)
            path_realtime = os.path.join(dirs["realtime"], f"{base_name}_display.png")
            cv2.imwrite(path_realtime, gabung)
            print(f"     realtime display: {path_realtime}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[SELESAI] Inference realtime selesai.")


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n" + "=" * 78)
    print("SCRIPT 10 — INFERENCE U-NET SEGMENTASI BATUBARA - SELARAS")
    print("=" * 78)
    print(f"Mode       : {MODE}")
    print(f"Model      : {MODEL_PATH}")
    print(f"Device     : {DEVICE}")
    print(f"Output dir : {OUTPUT_DIR}")
    print("=" * 78)

    model = load_model()

    if MODE == "folder":
        inference_folder(model)
    elif MODE == "camera":
        inference_camera(model)
    else:
        raise ValueError("MODE harus 'folder' atau 'camera'")

    print("\nOutput penting untuk tahap berikutnya:")
    print(f"- Mask binary 0/1 : {os.path.join(OUTPUT_DIR, 'masks_01')}")
    print(f"- Overlay cek visual: {os.path.join(OUTPUT_DIR, 'overlays')}")
    print("Mask ini nanti digabungkan dengan depth map dan ROI/homografi gerbong untuk perhitungan volume.")


if __name__ == "__main__":
    main()
