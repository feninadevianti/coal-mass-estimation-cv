"""
SCRIPT 15B — EVALUASI SEGMENTASI SEMANTIK U-NET BATUBARA
=========================================================

Tujuan:
- Mengevaluasi kualitas mask segmentasi U-Net secara terpisah dari stereo vision dan massa akhir.
- Membandingkan prediksi U-Net dengan ground truth mask dari dataset test.

Input:
1. Model terbaik:
   hasil_training_unet_colab/model_best.pth
   atau hasil_training_unet/model_best.pth
2. Dataset binary:
   dataset_unet_binary_roboflow_FIXED_01/test/images
   dataset_unet_binary_roboflow_FIXED_01/test/masks

Output:
hasil_evaluasi_unet/
    evaluasi_unet_per_image.csv
    ringkasan_evaluasi_unet.txt
    preview_overlay/
    grafik_iou_per_image.png
    grafik_dice_per_image.png
    grafik_precision_recall_f1.png

Metrik:
- IoU
- Dice/F1
- Pixel Accuracy
- Precision
- Recall
"""

import os
import csv
from glob import glob
from datetime import datetime

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False


# =========================================================
# KONFIGURASI
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH_CANDIDATES = [
    os.path.join(BASE_DIR, "hasil_training_unet_colab", "model_best.pth"),
    os.path.join(BASE_DIR, "hasil_training_unet", "model_best.pth"),
]

DATASET_ROOT = os.path.join(BASE_DIR, "dataset_unet_binary_roboflow_FIXED_01")
SPLIT = "test"
OUTPUT_DIR = os.path.join(BASE_DIR, "hasil_evaluasi_unet")

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
THRESHOLD = 0.5
BASE_CHANNELS = 32
SAVE_PREVIEW = True
SAVE_PREVIEW_MAX = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# MODEL U-NET
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
    for p in MODEL_PATH_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "model_best.pth tidak ditemukan. Letakkan model di salah satu lokasi berikut:\n"
        + "\n".join(MODEL_PATH_CANDIDATES)
    )


def load_model():
    global IMAGE_WIDTH, IMAGE_HEIGHT, THRESHOLD, BASE_CHANNELS
    path = pilih_model_path()
    print(f"[INFO] Load model: {path}")
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=DEVICE)

    IMAGE_WIDTH = int(ckpt.get("image_width", IMAGE_WIDTH))
    IMAGE_HEIGHT = int(ckpt.get("image_height", IMAGE_HEIGHT))
    THRESHOLD = float(ckpt.get("threshold", THRESHOLD))
    BASE_CHANNELS = int(ckpt.get("base_channels", BASE_CHANNELS))

    model = UNet(3, 1, BASE_CHANNELS).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("[OK] Model dimuat")
    print(f"     Device     : {DEVICE}")
    print(f"     Input size : {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    print(f"     Threshold  : {THRESHOLD}")
    return model


# =========================================================
# DATASET PAIRING
# =========================================================
def cari_pasangan_image_mask():
    images_dir = os.path.join(DATASET_ROOT, SPLIT, "images")
    masks_dir = os.path.join(DATASET_ROOT, SPLIT, "masks")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Folder images tidak ditemukan: {images_dir}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Folder masks tidak ditemukan: {masks_dir}")

    mask_files = sorted(glob(os.path.join(masks_dir, "*.png")))
    image_exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
    pairs = []

    for mask_path in mask_files:
        mask_name = os.path.basename(mask_path)
        base = os.path.splitext(mask_name)[0]
        kandidat = []
        if base.endswith("_mask"):
            base_img = base[:-5]
            kandidat += [base_img + ext for ext in image_exts]
        kandidat += [base + ext for ext in image_exts]

        image_path = None
        for k in kandidat:
            p = os.path.join(images_dir, k)
            if os.path.exists(p):
                image_path = p
                break

        if image_path is None:
            base_no_mask = base[:-5] if base.endswith("_mask") else base
            for fname in os.listdir(images_dir):
                if fname.startswith(base_no_mask) and os.path.splitext(fname)[1].lower() in image_exts:
                    image_path = os.path.join(images_dir, fname)
                    break

        if image_path is None:
            print(f"[WARNING] Image pasangan tidak ditemukan untuk mask: {mask_name}")
            continue
        pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError("Tidak ada pasangan image-mask untuk evaluasi.")
    return pairs


# =========================================================
# PREDIKSI & METRIK
# =========================================================
def preprocess(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0).to(DEVICE, dtype=torch.float32)


@torch.no_grad()
def prediksi(model, img_bgr, original_size):
    tensor = preprocess(img_bgr)
    logits = model(tensor)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    pred = (prob > THRESHOLD).astype(np.uint8)
    pred = cv2.resize(pred, original_size, interpolation=cv2.INTER_NEAREST)
    return pred


def baca_mask(path, size):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Gagal membaca mask: {path}")
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def hitung_metric(pred, gt, eps=1e-7):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "iou": float(iou),
        "dice": float(dice),
        "pixel_accuracy": float(pixel_acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
    }


# =========================================================
# PREVIEW
# =========================================================
def buat_overlay(img_bgr, mask, warna=(0, 255, 0)):
    color = np.zeros_like(img_bgr)
    color[mask.astype(bool)] = warna
    return cv2.addWeighted(img_bgr, 0.65, color, 0.35, 0)


def panel(img, title, w=320, h=240):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    header_h = 32
    body_h = h - header_h
    ih, iw = img.shape[:2]
    scale = min(w / iw, body_h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x0 = (w - nw) // 2
    y0 = header_h + (body_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    cv2.rectangle(canvas, (0, 0), (w, header_h), (35, 35, 35), -1)
    cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return canvas


def simpan_preview(img, gt, pred, metric, out_path):
    gt_vis = (gt * 255).astype(np.uint8)
    pred_vis = (pred * 255).astype(np.uint8)
    overlay_gt = buat_overlay(img, gt, warna=(255, 0, 0))
    overlay_pred = buat_overlay(img, pred, warna=(0, 255, 0))

    p1 = panel(img, "IMAGE")
    p2 = panel(gt_vis, "GT MASK")
    p3 = panel(pred_vis, "PRED MASK")
    p4 = panel(overlay_pred, f"IoU={metric['iou']:.3f} Dice={metric['dice']:.3f}")
    out = np.hstack([p1, p2, p3, p4])
    cv2.imwrite(out_path, out)


# =========================================================
# OUTPUT
# =========================================================
def pastikan_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_PREVIEW:
        os.makedirs(os.path.join(OUTPUT_DIR, "preview_overlay"), exist_ok=True)


def simpan_csv(rows):
    path = os.path.join(OUTPUT_DIR, "evaluasi_unet_per_image.csv")
    fieldnames = [
        "no", "image", "mask", "iou", "dice", "pixel_accuracy", "precision", "recall", "specificity",
        "tp", "tn", "fp", "fn"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def simpan_ringkasan(rows):
    path = os.path.join(OUTPUT_DIR, "ringkasan_evaluasi_unet.txt")
    metrics = ["iou", "dice", "pixel_accuracy", "precision", "recall", "specificity"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("RINGKASAN EVALUASI SEGMENTASI SEMANTIK U-NET\n")
        f.write("============================================\n")
        f.write(f"Waktu evaluasi : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset        : {DATASET_ROOT}\n")
        f.write(f"Split          : {SPLIT}\n")
        f.write(f"Jumlah data    : {len(rows)}\n")
        f.write(f"Threshold      : {THRESHOLD}\n")
        f.write("\nMETRIK RATA-RATA\n")
        for m in metrics:
            vals = np.array([float(r[m]) for r in rows], dtype=float)
            f.write(f"{m:16s}: mean={np.mean(vals):.6f} | median={np.median(vals):.6f} | min={np.min(vals):.6f} | max={np.max(vals):.6f}\n")
        f.write("\nCatatan: evaluasi ini menilai kualitas segmentasi U-Net, bukan depth atau massa akhir.\n")
    return path


def simpan_grafik(rows):
    if not MATPLOTLIB_OK:
        print("[WARNING] matplotlib tidak tersedia. Grafik tidak dibuat.")
        return []

    idx = np.arange(1, len(rows) + 1)
    iou = np.array([float(r["iou"]) for r in rows], dtype=float)
    dice = np.array([float(r["dice"]) for r in rows], dtype=float)
    precision = np.array([float(r["precision"]) for r in rows], dtype=float)
    recall = np.array([float(r["recall"]) for r in rows], dtype=float)

    paths = []
    path = os.path.join(OUTPUT_DIR, "grafik_iou_per_image.png")
    plt.figure(figsize=(9, 5))
    plt.bar(idx, iou)
    plt.xlabel("Data test")
    plt.ylabel("IoU")
    plt.title("IoU U-Net per Gambar")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    path = os.path.join(OUTPUT_DIR, "grafik_dice_per_image.png")
    plt.figure(figsize=(9, 5))
    plt.bar(idx, dice)
    plt.xlabel("Data test")
    plt.ylabel("Dice Score")
    plt.title("Dice Score U-Net per Gambar")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    path = os.path.join(OUTPUT_DIR, "grafik_precision_recall_f1.png")
    plt.figure(figsize=(9, 5))
    plt.plot(idx, precision, marker="o", label="Precision")
    plt.plot(idx, recall, marker="o", label="Recall")
    plt.plot(idx, dice, marker="o", label="Dice/F1")
    plt.xlabel("Data test")
    plt.ylabel("Nilai")
    plt.title("Precision, Recall, dan Dice/F1")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    return paths


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n" + "=" * 78)
    print("SCRIPT 15B — EVALUASI SEGMENTASI SEMANTIK U-NET BATUBARA")
    print("=" * 78)
    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Split        : {SPLIT}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Device       : {DEVICE}")
    print("=" * 78)

    pastikan_output()
    model = load_model()
    pairs = cari_pasangan_image_mask()
    print(f"[OK] Total pasangan image-mask: {len(pairs)}")

    rows = []
    for i, (img_path, mask_path) in enumerate(pairs, start=1):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARNING] Gagal baca image: {img_path}")
            continue
        h, w = img.shape[:2]
        gt = baca_mask(mask_path, (w, h))
        pred = prediksi(model, img, (w, h))
        metric = hitung_metric(pred, gt)

        row = {
            "no": i,
            "image": img_path,
            "mask": mask_path,
            **metric,
        }
        rows.append(row)

        if SAVE_PREVIEW and i <= SAVE_PREVIEW_MAX:
            out_path = os.path.join(OUTPUT_DIR, "preview_overlay", f"preview_unet_{i:03d}.png")
            simpan_preview(img, gt, pred, metric, out_path)

        print(f"[{i:03d}/{len(pairs)}] IoU={metric['iou']:.4f} Dice={metric['dice']:.4f} Acc={metric['pixel_accuracy']:.4f}")

    if not rows:
        raise RuntimeError("Tidak ada data yang berhasil dievaluasi.")

    csv_path = simpan_csv(rows)
    txt_path = simpan_ringkasan(rows)
    grafik_paths = simpan_grafik(rows)

    mean_iou = np.mean([float(r["iou"]) for r in rows])
    mean_dice = np.mean([float(r["dice"]) for r in rows])
    mean_acc = np.mean([float(r["pixel_accuracy"]) for r in rows])

    print("\nHASIL EVALUASI U-NET")
    print("-------------------")
    print(f"Mean IoU        : {mean_iou:.4f}")
    print(f"Mean Dice       : {mean_dice:.4f}")
    print(f"Mean Pixel Acc  : {mean_acc:.4f}")
    print("\n[OK] Output tersimpan")
    print(f"CSV       : {csv_path}")
    print(f"Ringkasan : {txt_path}")
    for p in grafik_paths:
        print(f"Grafik    : {p}")


if __name__ == "__main__":
    main()
