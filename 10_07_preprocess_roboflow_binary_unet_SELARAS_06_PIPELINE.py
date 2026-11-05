import os
import csv
import zipfile
import shutil
import cv2
import numpy as np
from datetime import datetime


# =========================================================
# KONFIGURASI
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_ZIP = os.path.join(BASE_DIR, "estimasi-muatan-batubara.v1i.png-mask-semantic.zip")
EXTRACT_DIR = os.path.join(BASE_DIR, "roboflow_export_unzipped")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_unet_binary_roboflow")

KALIBRASI_DIR = os.path.join(BASE_DIR, "hasil_kalibrasi")
STEREOMAP_XML = os.path.join(KALIBRASI_DIR, "stereomap.xml")
GUNAKAN_UKURAN_DARI_STEREOMAP = True

AUTO_CARI_ZIP = True

TARGET_WIDTH = 640
TARGET_HEIGHT = 480

COAL_PIXEL_VALUES = [1]

AUGMENT_TRAIN_ONLY = True

SAVE_OVERLAY_PREVIEW = True
MAX_PREVIEW_PER_SPLIT = 80

SAVE_MASK_AS_0_1 = True

IMAGE_EXT = ".png"

def buat_folder(path):
    os.makedirs(path, exist_ok=True)


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


def muat_target_size_dari_stereomap():
    global TARGET_WIDTH, TARGET_HEIGHT

    if not GUNAKAN_UKURAN_DARI_STEREOMAP:
        return None, None

    if not os.path.exists(STEREOMAP_XML):
        print("[WARNING] stereomap.xml tidak ditemukan. Target tetap memakai default 640x480.")
        return None, None

    fs = cv2.FileStorage(STEREOMAP_XML, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        print("[WARNING] stereomap.xml gagal dibuka. Target tetap memakai default 640x480.")
        return None, None

    image_width = baca_int_xml(fs, "image_width", default=TARGET_WIDTH)
    image_height = baca_int_xml(fs, "image_height", default=TARGET_HEIGHT)
    rms_stereo = baca_float_xml(fs, "rms_stereo", default=None)
    baseline_mm = baca_float_xml(fs, "baseline_mm", default=None)
    fs.release()

    TARGET_WIDTH = int(image_width)
    TARGET_HEIGHT = int(image_height)

    print("[OK] Target size diselaraskan dari stereomap.xml")
    print(f"     Target      : {TARGET_WIDTH} x {TARGET_HEIGHT}")
    if rms_stereo is not None:
        print(f"     RMS stereo  : {rms_stereo:.6f} px")
    if baseline_mm is not None:
        print(f"     Baseline    : {baseline_mm:.3f} mm | {baseline_mm/10.0:.3f} cm")

    return rms_stereo, baseline_mm


def cari_zip_otomatis():
    if os.path.exists(INPUT_ZIP):
        return INPUT_ZIP

    if not AUTO_CARI_ZIP:
        return INPUT_ZIP

    kandidat = [
        os.path.join(BASE_DIR, f)
        for f in os.listdir(BASE_DIR)
        if f.lower().endswith(".zip")
    ]

    if len(kandidat) == 1:
        print(f"[INFO] INPUT_ZIP default tidak ditemukan. Memakai ZIP otomatis: {os.path.basename(kandidat[0])}")
        return kandidat[0]

    if len(kandidat) > 1:
        print("[INFO] Ditemukan beberapa file ZIP di folder script:")
        for i, p in enumerate(kandidat, 1):
            print(f"  [{i}] {os.path.basename(p)}")
        while True:
            try:
                pilih = int(input("Pilih nomor ZIP Roboflow: "))
                if 1 <= pilih <= len(kandidat):
                    return kandidat[pilih - 1]
            except Exception:
                pass
            print("Pilihan tidak valid.")

    return INPUT_ZIP


def reset_folder_output():
    if os.path.exists(OUTPUT_DIR):
        print(f"[INFO] Menghapus output lama: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    buat_folder(OUTPUT_DIR)


def ekstrak_zip():
    input_zip_final = cari_zip_otomatis()

    if not os.path.exists(input_zip_final):
        raise FileNotFoundError(
            f"File ZIP tidak ditemukan: {input_zip_final}\n"
            "Letakkan file ZIP Roboflow di folder yang sama dengan script ini."
        )

    if os.path.exists(EXTRACT_DIR):
        print(f"[INFO] Menghapus folder ekstraksi lama: {EXTRACT_DIR}")
        shutil.rmtree(EXTRACT_DIR)

    buat_folder(EXTRACT_DIR)

    print(f"[INFO] Mengekstrak ZIP Roboflow:\n       {input_zip_final}")
    with zipfile.ZipFile(input_zip_final, "r") as z:
        z.extractall(EXTRACT_DIR)

    print(f"[OK] Dataset berhasil diekstrak ke:\n     {EXTRACT_DIR}")


def baca_classes_csv():
    path_csv = os.path.join(EXTRACT_DIR, "_classes.csv")
    if not os.path.exists(path_csv):
        print("[WARNING] _classes.csv tidak ditemukan.")
        return

    print("\n[INFO] Isi _classes.csv:")
    with open(path_csv, "r", encoding="utf-8", errors="replace") as f:
        for line in f.readlines():
            print("      " + line.strip())


def cari_split_folder():
    splits = []
    for split in ["train", "valid", "test"]:
        path = os.path.join(EXTRACT_DIR, split)
        if os.path.isdir(path):
            splits.append(split)

    if not splits:
        raise RuntimeError("Folder train/valid/test tidak ditemukan di hasil export Roboflow.")

    print(f"[OK] Split dataset ditemukan: {splits}")
    return splits


def cari_pasangan_image_mask(split):
    folder = os.path.join(EXTRACT_DIR, split)

    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    pairs = []

    for fname in sorted(os.listdir(folder)):
        low = fname.lower()

        if low.endswith("_mask.png"):
            continue

        if not any(low.endswith(ext) for ext in image_exts):
            continue

        base, _ = os.path.splitext(fname)
        mask_name = base + "_mask.png"
        mask_path = os.path.join(folder, mask_name)
        image_path = os.path.join(folder, fname)

        if os.path.exists(mask_path):
            pairs.append((image_path, mask_path, base))
        else:
            print(f"[WARNING] Mask tidak ditemukan untuk image: {fname}")

    print(f"[OK] {split}: {len(pairs)} pasangan image-mask")
    return pairs


def baca_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Gagal membaca image: {path}")
    return img


def baca_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Gagal membaca mask: {path}")

    # Jika mask terbaca 3 channel, ambil channel pertama.
    # Untuk PNG mask semantic Roboflow biasanya sudah 1 channel.
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return mask


def resize_image_mask(img, mask):
    img_resize = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    mask_resize = cv2.resize(mask, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)
    return img_resize, mask_resize


def konversi_mask_biner(mask):
    """
    
Output:
        mask_binary bernilai 0 dan 1.
    """
    mask_binary = np.isin(mask, COAL_PIXEL_VALUES).astype(np.uint8)

    if SAVE_MASK_AS_0_1:
        return mask_binary

    return (mask_binary * 255).astype(np.uint8)


def buat_overlay(img, mask_binary):
    """
    Overlay hijau pada area batubara.
    """
    if mask_binary.max() > 1:
        mask_vis = (mask_binary > 0).astype(np.uint8)
    else:
        mask_vis = mask_binary.astype(np.uint8)

    warna = np.zeros_like(img)
    warna[mask_vis == 1] = (0, 255, 0)

    overlay = cv2.addWeighted(img, 0.65, warna, 0.35, 0)
    return overlay


def nama_file_aman(teks):
    aman = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in teks)
    while "__" in aman:
        aman = aman.replace("__", "_")
    return aman.strip("_")


# =========================================================
# AUGMENTASI
# =========================================================
def tambah_brightness(img, beta):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)


def tambah_contrast(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def rotasi_image_mask(img, mask, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    img_rot = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    mask_rot = cv2.warpAffine(
        mask, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return img_rot, mask_rot


def buat_augmentasi(img, mask, split):
    """
    Augmentasi hanya untuk train.
    valid/test hanya original.
    """
    hasil = []

    hasil.append(("orig", img, mask))

    if AUGMENT_TRAIN_ONLY and split != "train":
        return hasil

    # Horizontal flip masih cukup aman untuk bentuk gerbong/batubara.
    hasil.append(("hflip", cv2.flip(img, 1), cv2.flip(mask, 1)))

    # Brightness/contrast tidak mengubah mask.
    hasil.append(("bright_plus", tambah_brightness(img, 25), mask.copy()))
    hasil.append(("bright_minus", tambah_brightness(img, -25), mask.copy()))
    hasil.append(("contrast_plus", tambah_contrast(img, 1.18), mask.copy()))
    hasil.append(("contrast_minus", tambah_contrast(img, 0.85), mask.copy()))

    # Rotasi kecil, mask memakai nearest supaya label tidak rusak.
    img_r1, mask_r1 = rotasi_image_mask(img, mask, 3)
    img_r2, mask_r2 = rotasi_image_mask(img, mask, -3)
    hasil.append(("rot_plus3", img_r1, mask_r1))
    hasil.append(("rot_minus3", img_r2, mask_r2))

    return hasil


# =========================================================
# SIMPAN OUTPUT
# =========================================================
def buat_output_split(split):
    dirs = {
        "images": os.path.join(OUTPUT_DIR, split, "images"),
        "masks": os.path.join(OUTPUT_DIR, split, "masks"),
        "masks_visual": os.path.join(OUTPUT_DIR, split, "masks_visual"),
        "preview_overlay": os.path.join(OUTPUT_DIR, split, "preview_overlay"),
    }

    for d in dirs.values():
        buat_folder(d)

    return dirs


def tulis_log_header(path_log):
    with open(path_log, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "split",
            "source_image",
            "source_mask",
            "augmentasi",
            "output_image",
            "output_mask_0_1",
            "output_mask_visual_0_255",
            "output_overlay",
            "original_mask_values",
            "binary_mask_values",
            "coal_pixel_count",
            "background_pixel_count",
            "target_size",
        ])


def tambah_log(path_log, row):
    with open(path_log, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def simpan_sample(split, dirs, base, aug_name, img, mask_binary, source_image, source_mask, original_values, preview_count):
    safe_base = nama_file_aman(base)
    out_base = f"{safe_base}_{aug_name}"

    image_name = out_base + IMAGE_EXT
    mask_name = out_base + "_mask.png"
    mask_visual_name = out_base + "_mask_visual.png"
    overlay_name = out_base + "_overlay.png"

    image_path = os.path.join(dirs["images"], image_name)
    mask_path = os.path.join(dirs["masks"], mask_name)
    mask_visual_path = os.path.join(dirs["masks_visual"], mask_visual_name)
    overlay_path = os.path.join(dirs["preview_overlay"], overlay_name)

    cv2.imwrite(image_path, img)

    # Mask training: tetap 0/1
    cv2.imwrite(mask_path, mask_binary.astype(np.uint8))

    # Mask visual: 0/255 agar terlihat jelas jika dibuka sebagai gambar biasa.
    mask_visual = (mask_binary.astype(np.uint8) * 255)
    cv2.imwrite(mask_visual_path, mask_visual)

    overlay_save = ""
    if SAVE_OVERLAY_PREVIEW and preview_count < MAX_PREVIEW_PER_SPLIT:
        overlay = buat_overlay(img, mask_binary)
        cv2.imwrite(overlay_path, overlay)
        overlay_save = overlay_path

    binary_values = np.unique(mask_binary).tolist()
    coal_count = int((mask_binary == 1).sum())
    background_count = int((mask_binary == 0).sum())

    return {
        "image_path": image_path,
        "mask_path": mask_path,
        "mask_visual_path": mask_visual_path,
        "overlay_path": overlay_save,
        "binary_values": binary_values,
        "coal_count": coal_count,
        "background_count": background_count,
    }


# =========================================================
# PROSES UTAMA
# =========================================================
def proses_dataset():
    rms_stereo, baseline_mm = muat_target_size_dari_stereomap()

    reset_folder_output()
    ekstrak_zip()
    baca_classes_csv()

    splits = cari_split_folder()

    path_log = os.path.join(OUTPUT_DIR, "preprocessing_log.csv")
    path_unique = os.path.join(OUTPUT_DIR, "unique_mask_values_report.csv")

    tulis_log_header(path_log)

    with open(path_unique, "w", newline="", encoding="utf-8") as f_unique:
        writer_unique = csv.writer(f_unique)
        writer_unique.writerow(["split", "mask_file", "unique_values", "counts"])

    total_output = 0
    total_coal_pixels = 0
    total_bg_pixels = 0

    print("\n" + "=" * 72)
    print("MULAI PREPROCESS DATASET ROBOFLOW -> BINARY U-NET")
    print("=" * 72)
    print(f"Mapping aktif    : pixel value {COAL_PIXEL_VALUES} -> batubara (1), selain itu -> background (0)")
    print(f"Resize target    : {TARGET_WIDTH} x {TARGET_HEIGHT}")
    print(f"Output folder    : {OUTPUT_DIR}")
    print("Sumber ideal     : export Roboflow dari dataset_unet/kiri_rectified")
    print("Remap/crop ulang : TIDAK, karena gambar sudah harus rectified")
    if rms_stereo is not None:
        print(f"RMS stereo       : {rms_stereo:.6f} px")
    if baseline_mm is not None:
        print(f"Baseline         : {baseline_mm:.3f} mm | {baseline_mm/10.0:.3f} cm")
    print("=" * 72)

    for split in splits:
        pairs = cari_pasangan_image_mask(split)
        dirs = buat_output_split(split)

        preview_count = 0
        split_output = 0

        for image_path, mask_path, base in pairs:
            img = baca_image(image_path)
            mask = baca_mask(mask_path)

            original_values, original_counts = np.unique(mask, return_counts=True)
            original_values_list = original_values.astype(int).tolist()
            original_counts_list = original_counts.astype(int).tolist()

            with open(path_unique, "a", newline="", encoding="utf-8") as f_unique:
                writer_unique = csv.writer(f_unique)
                writer_unique.writerow([
                    split,
                    os.path.basename(mask_path),
                    original_values_list,
                    original_counts_list
                ])

            img, mask = resize_image_mask(img, mask)

            # Ubah mask multi-class menjadi binary 0/1 sebelum augmentasi.
            mask_binary = konversi_mask_biner(mask)

            for aug_name, img_aug, mask_aug in buat_augmentasi(img, mask_binary, split):
                # Pastikan setelah augmentasi mask tetap 0/1.
                mask_aug = (mask_aug > 0).astype(np.uint8)

                saved = simpan_sample(
                    split=split,
                    dirs=dirs,
                    base=base,
                    aug_name=aug_name,
                    img=img_aug,
                    mask_binary=mask_aug,
                    source_image=image_path,
                    source_mask=mask_path,
                    original_values=original_values_list,
                    preview_count=preview_count,
                )

                if saved["overlay_path"]:
                    preview_count += 1

                total_coal_pixels += saved["coal_count"]
                total_bg_pixels += saved["background_count"]

                tambah_log(path_log, [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    split,
                    image_path,
                    mask_path,
                    aug_name,
                    saved["image_path"],
                    saved["mask_path"],
                    saved["mask_visual_path"],
                    saved["overlay_path"],
                    original_values_list,
                    saved["binary_values"],
                    saved["coal_count"],
                    saved["background_count"],
                    f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
                ])

                split_output += 1
                total_output += 1

        print(f"[OK] {split}: output tersimpan = {split_output}")

    print("\n" + "=" * 72)
    print("PREPROCESS SELESAI")
    print("=" * 72)
    print(f"Total output image-mask : {total_output}")
    print(f"Output dataset          : {OUTPUT_DIR}")
    print(f"Log preprocessing       : {path_log}")
    print(f"Report unique mask      : {path_unique}")
    print(f"Total piksel batubara   : {total_coal_pixels}")
    print(f"Total piksel background : {total_bg_pixels}")
    print("=" * 72)

    if total_coal_pixels == 0:
        print("\n[ERROR] PERINGATAN PENTING:")
        print("Tidak ada satu pun piksel batubara terdeteksi di seluruh dataset output!")
        print(f"COAL_PIXEL_VALUES saat ini = {COAL_PIXEL_VALUES}")
        print("Kemungkinan besar nilai pixel mask aktual TIDAK SESUAI dengan COAL_PIXEL_VALUES.")
        print(f"Cek ulang {path_unique} untuk melihat nilai unik mask yang sebenarnya,")
        print("lalu sesuaikan konstanta COAL_PIXEL_VALUES di bagian KONFIGURASI sebelum training.")

    print("\nCatatan penting:")
    print("Mask training di folder masks bernilai 0 dan 1.")
    print("Mask di folder masks_visual bernilai 0 dan 255 hanya untuk dilihat.")
    print("Overlay preview dipakai untuk mengecek apakah area batubara sudah benar.")


if __name__ == "__main__":
    proses_dataset()
