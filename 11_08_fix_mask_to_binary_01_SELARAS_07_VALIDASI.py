"""
SCRIPT 8 — VALIDASI DAN FIX MASK DATASET MENJADI BINARY 0/1
SELARAS SCRIPT 7 PREPROCESS ROBOFLOW + PIPELINE U-NET
=====================================================

Fungsi:
1. Mengecek folder dataset_unet_binary_roboflow.
2. Membaca semua mask di train/valid/test.
3. Memaksa semua mask menjadi binary:
   - Jika mask sudah 0/1 -> dibiarkan
   - Jika mask 0/255 -> diubah menjadi 0/1
   - Jika mask multi-value lain -> nilai di COAL_VALUES_MULTI_CLASS dianggap batubara,
     selebihnya dianggap background
4. Output disimpan ke folder baru:
   dataset_unet_binary_roboflow_FIXED_01

Gunakan folder hasil FIXED ini untuk training U-Net.

Catatan selaras pipeline:
- Input script ini adalah output Script 7: dataset_unet_binary_roboflow.
- Tahap ini hanya validasi/perbaikan mask 0/1; tidak melakukan crop, remap, atau homografi.
- Homografi dipakai nanti pada tahap ROI/volume gerbong, bukan pada tahap fix mask U-Net.
- Pada dataset 200 gambar (split 140/30/30) yang sudah diverifikasi, mask dari Script 7
  semestinya SUDAH 0/1 murni. Cabang COAL_VALUES_MULTI_CLASS di bawah ini hanya berfungsi
  sebagai pengaman/fallback jika suatu saat ada mask yang lolos dari Script 7 dalam
  bentuk multi-value mentah (belum dikonversi binary).
"""

import os
import shutil
import csv
import cv2
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_ROOT = os.path.join(BASE_DIR, "dataset_unet_binary_roboflow")
OUTPUT_ROOT = os.path.join(BASE_DIR, "dataset_unet_binary_roboflow_FIXED_01")

SPLITS = ["train", "valid", "test"]

# Mapping fallback jika ditemukan mask multi-value mentah (seharusnya jarang terjadi
# karena Script 7 sudah mengeluarkan mask 0/1):
# 1 = batubara (sesuai _classes.csv dataset 200 gambar)
# 0 = background
COAL_VALUES_MULTI_CLASS = [1]

SAVE_MASK_VISUAL_255 = True
SAVE_OVERLAY_PREVIEW = True

# Homografi tidak dipakai di tahap ini. Jangan warp mask di sini,
# karena training U-Net harus tetap konsisten dengan citra rectified dari Script 6/7.
GUNAKAN_HOMOGRAFI_DI_TAHAP_INI = False


def buat_folder(path):
    os.makedirs(path, exist_ok=True)


def unique_values(mask):
    return sorted(np.unique(mask).astype(int).tolist())


def force_binary_mask(mask):
    """
    Output wajib 0 dan 1.
    """
    vals = set(unique_values(mask))

    # Kasus benar: sudah 0 dan 1
    if vals.issubset({0, 1}):
        return mask.astype(np.uint8)

    # Kasus visual mask: 0 dan 255
    if vals.issubset({0, 255}):
        return (mask > 0).astype(np.uint8)

    # Kasus Roboflow semantic multi-value (fallback):
    # 0 = background
    # 1 = batubara (lihat COAL_VALUES_MULTI_CLASS)
    return np.isin(mask, COAL_VALUES_MULTI_CLASS).astype(np.uint8)


def cari_image_pasangan(images_dir, mask_name):
    """
    Mencari image pasangan dari nama mask.
    Contoh:
    abc_orig_mask.png -> abc_orig.png
    """
    kandidat = []

    if mask_name.endswith("_mask.png"):
        kandidat.append(mask_name.replace("_mask.png", ".png"))
        kandidat.append(mask_name.replace("_mask.png", ".jpg"))
        kandidat.append(mask_name.replace("_mask.png", ".jpeg"))

    base = os.path.splitext(mask_name)[0]
    kandidat += [
        base + ".png",
        base + ".jpg",
        base + ".jpeg",
    ]

    for nama in kandidat:
        path = os.path.join(images_dir, nama)
        if os.path.exists(path):
            return path

    return None


def buat_overlay(img, mask_binary):
    warna = np.zeros_like(img)
    warna[mask_binary == 1] = (0, 255, 0)
    return cv2.addWeighted(img, 0.65, warna, 0.35, 0)


def proses_split(split, writer):
    in_images = os.path.join(INPUT_ROOT, split, "images")
    in_masks = os.path.join(INPUT_ROOT, split, "masks")

    out_images = os.path.join(OUTPUT_ROOT, split, "images")
    out_masks = os.path.join(OUTPUT_ROOT, split, "masks")
    out_masks_visual = os.path.join(OUTPUT_ROOT, split, "masks_visual")
    out_overlay = os.path.join(OUTPUT_ROOT, split, "preview_overlay")

    if not os.path.isdir(in_masks):
        print(f"[SKIP] Folder mask tidak ditemukan: {in_masks}")
        return 0, 0, 0

    buat_folder(out_images)
    buat_folder(out_masks)
    buat_folder(out_masks_visual)
    buat_folder(out_overlay)

    # Copy semua image
    if os.path.isdir(in_images):
        for fname in os.listdir(in_images):
            src = os.path.join(in_images, fname)
            dst = os.path.join(out_images, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    total = 0
    total_coal_pixels = 0
    total_bg_pixels = 0

    for mask_name in sorted(os.listdir(in_masks)):
        if not mask_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path_mask = os.path.join(in_masks, mask_name)
        mask = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED)

        if mask is None:
            print(f"[WARNING] Mask gagal dibaca: {path_mask}")
            continue

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        before_vals = unique_values(mask)
        mask_bin = force_binary_mask(mask)
        after_vals = unique_values(mask_bin)

        out_mask_path = os.path.join(out_masks, mask_name)
        cv2.imwrite(out_mask_path, mask_bin.astype(np.uint8))

        out_visual_path = ""
        if SAVE_MASK_VISUAL_255:
            out_visual_path = os.path.join(out_masks_visual, mask_name)
            cv2.imwrite(out_visual_path, (mask_bin * 255).astype(np.uint8))

        overlay_path = ""
        if SAVE_OVERLAY_PREVIEW and os.path.isdir(in_images):
            img_path = cari_image_pasangan(in_images, mask_name)
            if img_path is not None:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    if img.shape[:2] != mask_bin.shape[:2]:
                        img = cv2.resize(img, (mask_bin.shape[1], mask_bin.shape[0]))
                    overlay = buat_overlay(img, mask_bin)
                    overlay_path = os.path.join(out_overlay, mask_name.replace("_mask", "_overlay"))
                    cv2.imwrite(overlay_path, overlay)

        coal_pixels = int((mask_bin == 1).sum())
        bg_pixels = int((mask_bin == 0).sum())
        total_coal_pixels += coal_pixels
        total_bg_pixels += bg_pixels

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            split,
            path_mask,
            out_mask_path,
            before_vals,
            after_vals,
            coal_pixels,
            bg_pixels,
            out_visual_path,
            overlay_path,
        ])

        if after_vals != [0] and after_vals != [0, 1]:
            print(f"[ERROR] Mask belum binary: {out_mask_path} -> {after_vals}")

        total += 1

    print(f"[OK] {split}: {total} mask dikonversi ke binary 0/1")
    print(f"     Piksel batubara total : {total_coal_pixels}")
    print(f"     Piksel background total: {total_bg_pixels}")
    return total, total_coal_pixels, total_bg_pixels


def main():
    if not os.path.isdir(INPUT_ROOT):
        raise FileNotFoundError(
            f"Folder input tidak ditemukan: {INPUT_ROOT}\n"
            "Jalankan dulu script 07_preprocess_roboflow_binary_unet_SELARAS.py"
        )

    if os.path.exists(OUTPUT_ROOT):
        print(f"[INFO] Menghapus output lama: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    buat_folder(OUTPUT_ROOT)

    log_path = os.path.join(OUTPUT_ROOT, "fix_mask_binary_log.csv")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "split",
            "input_mask",
            "output_mask",
            "unique_before",
            "unique_after",
            "coal_pixels",
            "background_pixels",
            "mask_visual_255",
            "overlay_preview",
        ])

        total_all = 0
        total_coal_all = 0
        total_bg_all = 0
        for split in SPLITS:
            total_split, coal_split, bg_split = proses_split(split, writer)
            total_all += total_split
            total_coal_all += coal_split
            total_bg_all += bg_split

    print("\n" + "=" * 70)
    print("SELESAI FIX MASK")
    print("=" * 70)
    print(f"Total mask diproses     : {total_all}")
    print(f"Total piksel batubara   : {total_coal_all}")
    print(f"Total piksel background: {total_bg_all}")
    print(f"Output dataset          : {OUTPUT_ROOT}")
    print(f"Log                     : {log_path}")
    print("=" * 70)

    if total_coal_all == 0:
        print("\n[ERROR] PERINGATAN PENTING:")
        print("Tidak ada satu pun piksel batubara terdeteksi di seluruh dataset FIXED!")
        print(f"COAL_VALUES_MULTI_CLASS saat ini = {COAL_VALUES_MULTI_CLASS}")
        print("Cek isi fix_mask_binary_log.csv kolom unique_before untuk melihat nilai")
        print("mask sebenarnya sebelum dipaksa biner, lalu sesuaikan konstanta di atas.")
        print("JANGAN lanjut training dengan dataset ini sebelum masalah ini diperbaiki.")

    print("\nGunakan folder ini untuk training U-Net:")
    print(OUTPUT_ROOT)
    print("\nFolder masks di output sudah bernilai 0 dan 1.")


if __name__ == "__main__":
    main()
