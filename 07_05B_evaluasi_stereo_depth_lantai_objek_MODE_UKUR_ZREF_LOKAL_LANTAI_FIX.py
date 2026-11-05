import os
import sys
import csv
from glob import glob
from datetime import datetime

import numpy as np

try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KALIBRASI_DIR = os.path.join(BASE_DIR, "hasil_kalibrasi")
SNAPSHOT_DIR = os.path.join(KALIBRASI_DIR, "snapshot_depth")
ZREF_NPZ = os.path.join(KALIBRASI_DIR, "zref.npz")

OUTPUT_DIR = os.path.join(BASE_DIR, "hasil_evaluasi_stereo_05b_mode_ukur")
TEMPLATE_CSV = os.path.join(OUTPUT_DIR, "data_evaluasi_stereo_05b_mode_ukur.csv")
HASIL_CSV = os.path.join(OUTPUT_DIR, "hasil_evaluasi_stereo_05b_mode_ukur.csv")
RINGKASAN_TXT = os.path.join(OUTPUT_DIR, "ringkasan_evaluasi_stereo_05b_mode_ukur.txt")
PREVIEW_DIR = os.path.join(OUTPUT_DIR, "preview_roi")

CENTER_CROP_LANTAI = 0.50
CENTER_CROP_OBJEK_DARURAT = 0.12  # hanya fallback kalau user belum pilih ROI manual

STAT_UTAMA = "median"

DEPTH_MIN_CM = 20.0
DEPTH_MAX_CM = 90.0

MIN_VALID_PIXELS = 30
MIN_VALID_ROI_LANTAI_PCT = 70.0
MIN_VALID_ROI_OBJEK_PCT = 50.0

BATAS_ERROR_CM_OK = 1.0
BATAS_ERROR_CM_CUKUP = 1.5

PAKAI_ZREF_VERTICAL = False

PAKAI_ZREF_LOKAL_DARI_ROI_LANTAI = True

def pastikan_folder():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)


def cari_snapshot_npz():
    files = sorted(glob(os.path.join(SNAPSHOT_DIR, "depth_snap_*.npz")), key=os.path.getmtime)
    return files


def baca_float_teks(x, default=None):
    if x is None:
        return default
    s = str(x).strip()
    if s == "":
        return default
    try:
        return float(s.replace(",", "."))
    except Exception:
        return default


def baca_int_teks(x, default=None):
    v = baca_float_teks(x, default=None)
    if v is None or not np.isfinite(v):
        return default
    return int(round(v))


def normalisasi(s):
    return str(s or "").strip().lower()


def baca_scalar_npz(npz, keys, default=np.nan):
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if key in npz:
            try:
                arr = np.asarray(npz[key]).astype(float).ravel()
                if arr.size > 0:
                    return float(arr[0])
            except Exception:
                pass
    return float(default)


def baca_string_npz(npz, keys, default=""):
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if key in npz:
            try:
                arr = np.asarray(npz[key]).ravel()
                if arr.size > 0:
                    val = arr[0]
                    if isinstance(val, bytes):
                        return val.decode("utf-8", errors="ignore")
                    return str(val)
            except Exception:
                pass
    return default


def muat_zref_global():
    if not os.path.exists(ZREF_NPZ):
        return np.nan, np.nan, np.nan, np.nan
    data = np.load(ZREF_NPZ, allow_pickle=True)
    zref_m = baca_scalar_npz(data, ["zref_m", "zref"], default=np.nan)
    zref_cm = baca_scalar_npz(data, ["zref_cm"], default=np.nan)
    zref_vertical_m = baca_scalar_npz(data, ["zref_vertical_m", "zref_vertical"], default=np.nan)
    zref_vertical_cm = baca_scalar_npz(data, ["zref_vertical_cm"], default=np.nan)

    if not np.isfinite(zref_cm) and np.isfinite(zref_m):
        zref_cm = zref_m * 100.0
    if not np.isfinite(zref_vertical_cm) and np.isfinite(zref_vertical_m):
        zref_vertical_cm = zref_vertical_m * 100.0
    return zref_m, zref_cm, zref_vertical_m, zref_vertical_cm


def muat_depth_npz(path):
    data = np.load(path, allow_pickle=True)

    if "depth_map" in data:
        depth_m = np.asarray(data["depth_map"], dtype=np.float32)
    elif "depth_m" in data:
        depth_m = np.asarray(data["depth_m"], dtype=np.float32)
    else:
        raise KeyError(f"File {path} tidak memiliki depth_map atau depth_m")

    if "valid_mask" in data:
        valid = np.asarray(data["valid_mask"]).astype(bool)
    elif "valid_depth" in data:
        valid = np.asarray(data["valid_depth"]).astype(bool)
    else:
        valid = np.isfinite(depth_m)

    snapshot_type = baca_string_npz(data, ["snapshot_type", "jenis_snapshot"], default="")
    snapshot_label = baca_string_npz(data, ["snapshot_label"], default="")
    snapshot_info = baca_string_npz(data, ["snapshot_info"], default="")

    zref_m_snap = baca_scalar_npz(data, ["zref", "zref_m"], default=np.nan)
    zref_vertical_m_snap = baca_scalar_npz(data, ["zref_vertical", "zref_vertical_m"], default=np.nan)
    zref_m_global, zref_cm_global, zref_vertical_m_global, zref_vertical_cm_global = muat_zref_global()

    zref_m = zref_m_snap if np.isfinite(zref_m_snap) else zref_m_global
    zref_cm = zref_m * 100.0 if np.isfinite(zref_m) else zref_cm_global

    zref_vertical_m = zref_vertical_m_snap if np.isfinite(zref_vertical_m_snap) else zref_vertical_m_global
    zref_vertical_cm = zref_vertical_m * 100.0 if np.isfinite(zref_vertical_m) else zref_vertical_cm_global

    return {
        "depth_m": depth_m,
        "valid_mask": valid,
        "zref_m": zref_m,
        "zref_cm": zref_cm,
        "zref_vertical_m": zref_vertical_m,
        "zref_vertical_cm": zref_vertical_cm,
        "snapshot_type": snapshot_type,
        "snapshot_label": snapshot_label,
        "snapshot_info": snapshot_info,
    }


def prefix(path):
    return os.path.splitext(os.path.basename(path))[0]


def center_roi(shape, ratio):
    h, w = shape[:2]
    ratio = max(0.03, min(1.0, float(ratio)))
    x1 = int(round(w * (0.5 - ratio / 2)))
    x2 = int(round(w * (0.5 + ratio / 2)))
    y1 = int(round(h * (0.5 - ratio / 2)))
    y2 = int(round(h * (0.5 + ratio / 2)))
    return x1, y1, x2, y2


def jenis_dari_snapshot(snapshot_type, urutan):
    st = str(snapshot_type or "").upper().strip()
    if st == "LANTAI_KOSONG":
        return "jarak_lantai"
    if st == "OBJEK_BALOK":
        return "tinggi_objek"
    if st == "MUATAN_BATUBARA":
        return "lewati"
    # fallback untuk snapshot lama tanpa metadata
    return "jarak_lantai" if urutan == 1 else "tinggi_objek"


def area_default_dari_jenis(jenis):
    if jenis == "jarak_lantai":
        return "center_crop_50"
    if jenis == "tinggi_objek":
        return "roi_manual_wajib"
    return "lewati"

# =========================================================
# ROI DAN STATISTIK
# =========================================================
def buat_roi_mask(shape, row, jenis_uji):
    h, w = shape[:2]
    area = normalisasi(row.get("area_pengukuran", "auto"))

    x1 = baca_int_teks(row.get("roi_x1"), None)
    y1 = baca_int_teks(row.get("roi_y1"), None)
    x2 = baca_int_teks(row.get("roi_x2"), None)
    y2 = baca_int_teks(row.get("roi_y2"), None)

    # ROI manual valid jika keempat koordinat terisi.
    if None not in (x1, y1, x2, y2):
        x1, x2 = sorted([max(0, min(w - 1, x1)), max(0, min(w, x2))])
        y1, y2 = sorted([max(0, min(h - 1, y1)), max(0, min(h, y2))])
        if (x2 - x1) >= 2 and (y2 - y1) >= 2:
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            return mask, (x1, y1, x2, y2), "roi_manual", ""

    if jenis_uji == "tinggi_objek":
        # Fallback kecil di tengah supaya script tetap bisa berjalan,
        # tetapi hasilnya diberi warning karena tinggi balok idealnya memakai ROI manual.
        x1, y1, x2, y2 = center_roi((h, w), CENTER_CROP_OBJEK_DARURAT)
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return mask, (x1, y1, x2, y2), f"center_crop_{int(CENTER_CROP_OBJEK_DARURAT*100)}_DARURAT", "ROI objek belum manual; sebaiknya pilih ROI permukaan atas balok dengan --pilih-roi"

    if area in {"full_valid", "full", "semua"}:
        mask = np.ones((h, w), dtype=bool)
        return mask, (0, 0, w, h), "full_valid", ""

    # Default lantai: 50% tengah.
    x1, y1, x2, y2 = center_roi((h, w), CENTER_CROP_LANTAI)
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask, (x1, y1, x2, y2), f"center_crop_{int(CENTER_CROP_LANTAI*100)}", ""


def hitung_stat_depth(depth_m, valid_mask, roi_mask):
    depth_cm = depth_m.astype(np.float32) * 100.0
    valid = (
        roi_mask &
        valid_mask.astype(bool) &
        np.isfinite(depth_cm) &
        (depth_cm >= DEPTH_MIN_CM) &
        (depth_cm <= DEPTH_MAX_CM)
    )
    vals = depth_cm[valid]
    total_roi = int(np.sum(roi_mask))
    n_valid = int(vals.size)
    pct_valid_roi = 100.0 * n_valid / total_roi if total_roi > 0 else 0.0

    if n_valid == 0:
        return {
            "n_valid": 0,
            "pct_valid_roi": pct_valid_roi,
            "depth_mean_cm": np.nan,
            "depth_median_cm": np.nan,
            "depth_std_cm": np.nan,
            "depth_p25_cm": np.nan,
            "depth_p75_cm": np.nan,
        }

    return {
        "n_valid": n_valid,
        "pct_valid_roi": pct_valid_roi,
        "depth_mean_cm": float(np.nanmean(vals)),
        "depth_median_cm": float(np.nanmedian(vals)),
        "depth_std_cm": float(np.nanstd(vals)),
        "depth_p25_cm": float(np.nanpercentile(vals, 25)),
        "depth_p75_cm": float(np.nanpercentile(vals, 75)),
    }


def nilai_depth_utama(stat):
    return stat["depth_mean_cm"] if STAT_UTAMA == "mean" else stat["depth_median_cm"]


def status_hasil(jenis, n_valid, valid_pct, error_abs, nilai_sistem_cm, zref_cm, depth_objek_cm, roi_mode, roi_warning):
    if n_valid < MIN_VALID_PIXELS:
        return "TIDAK VALID - piksel depth ROI kurang"
    if jenis == "jarak_lantai" and valid_pct < MIN_VALID_ROI_LANTAI_PCT:
        return "PERLU CEK - validROI lantai rendah"
    if jenis == "tinggi_objek" and valid_pct < MIN_VALID_ROI_OBJEK_PCT:
        return "PERLU CEK - validROI objek rendah"
    if not np.isfinite(error_abs):
        return "TIDAK VALID - nilai sistem tidak terbaca"
    if jenis == "tinggi_objek":
        if roi_warning:
            return "PERLU CEK - ROI objek belum manual"
        if np.isfinite(nilai_sistem_cm) and nilai_sistem_cm < 1.0:
            return "PERLU CEK - ROI kemungkinan masih lantai, bukan atas balok"
        if np.isfinite(nilai_sistem_cm) and nilai_sistem_cm < 0:
            return "TIDAK VALID - depth objek lebih jauh dari Z_ref"
    if error_abs <= BATAS_ERROR_CM_OK:
        return "VALID"
    if error_abs <= BATAS_ERROR_CM_CUKUP:
        return "CUKUP - masih layak prototype, cek ulang ROI"
    return "PERLU CEK - error > batas"

# =========================================================
# PREVIEW
# =========================================================
def depth_to_preview(depth_m, valid_mask):
    depth_cm = depth_m.astype(np.float32) * 100.0
    valid = valid_mask.astype(bool) & np.isfinite(depth_cm) & (depth_cm >= DEPTH_MIN_CM) & (depth_cm <= DEPTH_MAX_CM)
    out = np.zeros(depth_cm.shape, dtype=np.uint8)
    if np.any(valid):
        v = depth_cm[valid]
        lo, hi = np.nanpercentile(v, [3, 97])
        if hi <= lo:
            hi = lo + 1.0
        norm = (depth_cm - lo) / (hi - lo)
        norm = np.clip(norm, 0, 1)
        norm[~valid] = 0
        out = (norm * 255).astype(np.uint8)
    if CV2_OK:
        cm = cv2.applyColorMap(out, cv2.COLORMAP_TURBO)
        cm[~valid] = (25, 25, 25)
        return cm
    return np.dstack([out, out, out])


def simpan_preview(path_npz, depth_m, valid_mask, roi_tuple, label, nilai_sistem_cm=None, aktual_cm=None, depth_roi_cm=None, zref_cm=None, status=""):
    if not CV2_OK:
        return ""
    img = depth_to_preview(depth_m, valid_mask)
    x1, y1, x2, y2 = roi_tuple
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.rectangle(img, (0, 0), (img.shape[1], 92), (0, 0, 0), -1)
    cv2.putText(img, label[:80], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    teks2 = []
    if depth_roi_cm is not None and np.isfinite(depth_roi_cm):
        teks2.append(f"depthROI={depth_roi_cm:.2f}cm")
    if zref_cm is not None and np.isfinite(zref_cm):
        teks2.append(f"Zref={zref_cm:.2f}cm")
    if nilai_sistem_cm is not None and np.isfinite(nilai_sistem_cm):
        teks2.append(f"sistem={nilai_sistem_cm:.2f}cm")
    if aktual_cm is not None and np.isfinite(aktual_cm):
        teks2.append(f"aktual={aktual_cm:.2f}cm")
    cv2.putText(img, " | ".join(teks2)[:95], (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
    cv2.putText(img, status[:95], (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0) if status.startswith("VALID") else (0, 180, 255), 1)

    out = os.path.join(PREVIEW_DIR, f"preview_{prefix(path_npz)}.png")
    cv2.imwrite(out, img)
    return out

# =========================================================
# TEMPLATE CSV
# =========================================================
def buat_template(force=False):
    pastikan_folder()
    snaps = cari_snapshot_npz()
    if not snaps:
        print("\n[ERROR] Belum ada snapshot depth dari Script 05.")
        print(f"Folder yang dicek: {SNAPSHOT_DIR}")
        print("Jalankan Script 05 terbaru, lalu tekan S untuk menyimpan snapshot depth.")
        return False

    if os.path.exists(TEMPLATE_CSV) and not force:
        return True

    fieldnames = [
        "no", "snapshot_npz", "snapshot_type", "snapshot_label",
        "jenis_uji", "nilai_aktual_cm", "area_pengukuran",
        "roi_x1", "roi_y1", "roi_x2", "roi_y2", "catatan"
    ]
    rows = []
    for i, p in enumerate(snaps, start=1):
        try:
            meta = muat_depth_npz(p)
            snapshot_type = str(meta.get("snapshot_type", "")).upper().strip()
            snapshot_label = str(meta.get("snapshot_label", "")).strip()
        except Exception:
            snapshot_type = ""
            snapshot_label = ""
        jenis = jenis_dari_snapshot(snapshot_type, i)
        area = area_default_dari_jenis(jenis)
        if jenis == "jarak_lantai":
            catatan = "isi jarak aktual kamera-ke-lantai; ROI default center_crop_50"
        elif jenis == "tinggi_objek":
            catatan = "isi tinggi balok; WAJIB/SEBAIKNYA pilih ROI manual permukaan atas balok"
        else:
            catatan = "snapshot muatan batubara tidak dievaluasi di 05B"
        rows.append({
            "no": i,
            "snapshot_npz": p,
            "snapshot_type": snapshot_type,
            "snapshot_label": snapshot_label,
            "jenis_uji": jenis,
            "nilai_aktual_cm": "",
            "area_pengukuran": area,
            "roi_x1": "", "roi_y1": "", "roi_x2": "", "roi_y2": "",
            "catatan": catatan,
        })

    with open(TEMPLATE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tulis_panduan()
    print("\n[OK] Template evaluasi 05B mode ukur berhasil dibuat.")
    print(f"File: {TEMPLATE_CSV}")
    print("\nIsi kolom nilai_aktual_cm:")
    print("- LANTAI_KOSONG / jarak_lantai: isi jarak kamera ke lantai gerbong, contoh 56")
    print("- OBJEK_BALOK / tinggi_objek: isi tinggi balok, contoh 5")
    print("\nSetelah itu jalankan: python 05B...py --pilih-roi")
    return True


def tulis_panduan():
    teks = f"""PANDUAN EVALUASI STEREO 05B — MODE UKUR
=======================================

File CSV:
{TEMPLATE_CSV}

Makna jenis_uji:
1. jarak_lantai
   - berasal dari snapshot LANTAI_KOSONG.
   - nilai_aktual_cm = jarak manual kamera ke lantai gerbong.
   - ROI default = center_crop_50.

2. tinggi_objek
   - berasal dari snapshot OBJEK_BALOK.
   - nilai_aktual_cm = tinggi balok acuan, misalnya 5 cm.
   - ROI harus dipilih manual pada permukaan atas balok.
   - Jalankan: python {os.path.basename(__file__)} --pilih-roi

3. lewati
   - biasanya snapshot MUATAN_BATUBARA.
   - tidak dievaluasi di Script 05B.

Rumus:
- jarak_lantai = median depth ROI lantai.
- tinggi_objek = Z_ref - median depth ROI permukaan atas balok.

Jika hasil tinggi balok mendekati 0 cm:
- ROI kemungkinan masih berada di lantai/area gerbong besar, bukan di atas balok.
- Ulangi --pilih-roi dan pilih kotak kecil hanya pada permukaan atas balok.
"""
    with open(os.path.join(OUTPUT_DIR, "panduan_pengisian_evaluasi_05b_mode_ukur.txt"), "w", encoding="utf-8") as f:
        f.write(teks)


def baca_rows_csv():
    if not os.path.exists(TEMPLATE_CSV):
        return []
    with open(TEMPLATE_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tulis_rows_csv(rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(TEMPLATE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# =========================================================
# PILIH ROI MANUAL
# =========================================================
def mode_pilih_roi():
    if not CV2_OK:
        print("[ERROR] OpenCV tidak tersedia, mode pilih ROI tidak bisa dipakai.")
        return
    if "--reset-template" in sys.argv:
        buat_template(force=True)
    elif not buat_template():
        return

    rows = baca_rows_csv()
    if not rows:
        print("[ERROR] CSV template kosong.")
        return

    print("\nMODE PILIH ROI MANUAL 05B")
    print("Untuk balok: pilih kotak kecil hanya di permukaan atas balok.")
    print("Jangan kena lantai, sisi balok, dinding gerbong, atau bayangan.")
    print("Tekan ENTER/SPACE untuk simpan ROI, tekan C untuk skip.")

    for row in rows:
        jenis = normalisasi(row.get("jenis_uji", ""))
        if jenis == "lewati":
            continue
        p = row.get("snapshot_npz", "")
        if not os.path.exists(p):
            print(f"[WARNING] Snapshot tidak ditemukan: {p}")
            continue
        meta = muat_depth_npz(p)
        depth_m = meta["depth_m"]
        valid_mask = meta["valid_mask"]
        preview = depth_to_preview(depth_m, valid_mask)
        display = preview.copy()

        label = f"{row.get('no','')} | {os.path.basename(p)} | {jenis} | {row.get('snapshot_type','')}"
        cv2.rectangle(display, (0, 0), (display.shape[1], 72), (0, 0, 0), -1)
        cv2.putText(display, label[:85], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        if jenis == "tinggi_objek":
            instr = "PILIH ROI ATAS BALOK SAJA lalu ENTER/SPACE. C=skip."
        else:
            instr = "PILIH ROI LANTAI DATAR lalu ENTER/SPACE. C=skip."
        cv2.putText(display, instr, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,255,255), 1)

        # Tampilkan default ROI sebagai panduan sebelum user memilih.
        try:
            roi_mask, roi_tuple, roi_mode, _ = buat_roi_mask(depth_m.shape, row, jenis)
            x1, y1, x2, y2 = roi_tuple
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 180, 255), 1)
        except Exception:
            pass

        roi = cv2.selectROI("Pilih ROI Evaluasi 05B Mode Ukur", display, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Pilih ROI Evaluasi 05B Mode Ukur")
        x, y, ww, hh = [int(v) for v in roi]
        if ww <= 0 or hh <= 0:
            print(f"[SKIP] ROI tidak dipilih untuk {os.path.basename(p)}")
            continue

        row["area_pengukuran"] = "roi_manual"
        row["roi_x1"] = x
        row["roi_y1"] = y
        row["roi_x2"] = x + ww
        row["roi_y2"] = y + hh
        print(f"[OK] ROI {os.path.basename(p)}: x1={x}, y1={y}, x2={x+ww}, y2={y+hh}")

    tulis_rows_csv(rows)
    print(f"\n[OK] ROI tersimpan ke CSV: {TEMPLATE_CSV}")
    print("Isi nilai_aktual_cm jika belum diisi, lalu jalankan ulang script tanpa --pilih-roi.")

# =========================================================
# Z_REF LOKAL DARI SNAPSHOT LANTAI
# =========================================================
def hitung_zref_lokal_dari_lantai(rows):
    """
    Mengambil Z_ref lokal dari snapshot LANTAI_KOSONG/jarak_lantai.
    Nilai ini adalah median depth pada ROI lantai yang sama dengan evaluasi 05B.
    Untuk tinggi balok, ini sering lebih akurat daripada zref.npz global jika
    ROI Z_ref di Script 05 terlalu luas atau tidak tepat di lantai gerbong.
    """
    kandidat = []
    detail = []
    for row in rows:
        jenis = normalisasi(row.get("jenis_uji", ""))
        if jenis != "jarak_lantai":
            continue
        p = row.get("snapshot_npz", "")
        if not os.path.exists(p):
            continue
        actual_cm = baca_float_teks(row.get("nilai_aktual_cm"), None)
        try:
            meta = muat_depth_npz(p)
            depth_m = meta["depth_m"]
            valid_mask = meta["valid_mask"]
            roi_mask, roi_tuple, roi_mode, roi_warning = buat_roi_mask(depth_m.shape, row, jenis)
            stat = hitung_stat_depth(depth_m, valid_mask, roi_mask)
            depth_cm = nilai_depth_utama(stat)
            if not np.isfinite(depth_cm):
                continue
            if stat["n_valid"] < MIN_VALID_PIXELS:
                continue
            if stat["pct_valid_roi"] < MIN_VALID_ROI_LANTAI_PCT:
                continue
            kandidat.append(depth_cm)
            detail.append({
                "snapshot": os.path.basename(p),
                "depth_cm": depth_cm,
                "actual_cm": actual_cm,
                "valid_roi": stat["pct_valid_roi"],
                "roi_mode": roi_mode,
            })
        except Exception as e:
            print(f"[WARNING] Gagal hitung Z_ref lokal dari {os.path.basename(p)}: {e}")
            continue

    if not kandidat:
        return np.nan, []
    return float(np.nanmedian(np.asarray(kandidat, dtype=float))), detail

# =========================================================
# EVALUASI
# =========================================================
def proses_evaluasi():
    pastikan_folder()
    force = "--reset-template" in sys.argv
    if not buat_template(force=force):
        return

    rows = baca_rows_csv()
    if not rows:
        print("[ERROR] CSV kosong.")
        return

    _, zref_cm_global, _, zref_vertical_cm_global = muat_zref_global()
    zref_dipakai_global = zref_vertical_cm_global if PAKAI_ZREF_VERTICAL else zref_cm_global
    if not np.isfinite(zref_dipakai_global):
        print("\n[PERINGATAN] zref.npz belum ditemukan atau Z_ref tidak terbaca.")
        print(f"Dicek: {ZREF_NPZ}")
        print("Untuk tinggi_objek, jalankan Script 05 pada gerbong kosong lalu tekan Z dulu.")

    zref_lokal_lantai_cm, detail_zref_lokal = hitung_zref_lokal_dari_lantai(rows)
    if PAKAI_ZREF_LOKAL_DARI_ROI_LANTAI and np.isfinite(zref_lokal_lantai_cm):
        print(f"\n[INFO] Z_ref lokal dari ROI LANTAI_KOSONG dipakai untuk tinggi_objek: {zref_lokal_lantai_cm:.2f} cm")
        for d in detail_zref_lokal:
            print(f"       - {d['snapshot']} | depth lantai ROI={d['depth_cm']:.2f} cm | validROI={d['valid_roi']:.1f}% | ROI={d['roi_mode']}")
    elif PAKAI_ZREF_LOKAL_DARI_ROI_LANTAI:
        print("\n[INFO] Z_ref lokal lantai belum tersedia; tinggi_objek akan memakai zref.npz global.")

    hasil = []
    ada_nilai = False

    for row in rows:
        jenis = normalisasi(row.get("jenis_uji", ""))
        if jenis == "lewati":
            continue
        if jenis not in {"jarak_lantai", "tinggi_objek"}:
            print(f"[WARNING] jenis_uji tidak dikenal: {jenis}. Baris dilewati.")
            continue

        p = row.get("snapshot_npz", "")
        if not os.path.exists(p):
            print(f"[WARNING] Snapshot tidak ditemukan: {p}")
            continue

        actual_cm = baca_float_teks(row.get("nilai_aktual_cm"), None)
        if actual_cm is None:
            continue
        ada_nilai = True

        meta = muat_depth_npz(p)
        depth_m = meta["depth_m"]
        valid_mask = meta["valid_mask"]
        zref_cm_snap = meta["zref_vertical_cm"] if PAKAI_ZREF_VERTICAL else meta["zref_cm"]
        zref_cm_global_atau_snap = zref_cm_snap if np.isfinite(zref_cm_snap) else zref_dipakai_global
        zref_source_detail = "snapshot/global_zref"
        zref_cm = zref_cm_global_atau_snap
        if jenis == "tinggi_objek" and PAKAI_ZREF_LOKAL_DARI_ROI_LANTAI and np.isfinite(zref_lokal_lantai_cm):
            zref_cm = zref_lokal_lantai_cm
            zref_source_detail = "roi_lantai_lokal_05b"

        roi_mask, roi_tuple, roi_mode, roi_warning = buat_roi_mask(depth_m.shape, row, jenis)
        stat = hitung_stat_depth(depth_m, valid_mask, roi_mask)
        depth_utama_cm = nilai_depth_utama(stat)

        if jenis == "jarak_lantai":
            nilai_sistem_cm = depth_utama_cm
            jenis_label = "Jarak kamera ke lantai gerbong"
        else:
            jenis_label = "Tinggi objek acuan/balok"
            if np.isfinite(zref_cm) and np.isfinite(depth_utama_cm):
                nilai_sistem_cm = zref_cm - depth_utama_cm
            else:
                nilai_sistem_cm = np.nan

        error_cm = nilai_sistem_cm - actual_cm if np.isfinite(nilai_sistem_cm) else np.nan
        error_abs_cm = abs(error_cm) if np.isfinite(error_cm) else np.nan
        error_pct = (error_abs_cm / actual_cm * 100.0) if actual_cm and np.isfinite(error_abs_cm) and abs(actual_cm) > 1e-9 else np.nan
        status = status_hasil(jenis, stat["n_valid"], stat["pct_valid_roi"], error_abs_cm, nilai_sistem_cm, zref_cm, depth_utama_cm, roi_mode, roi_warning)

        label = f"{jenis_label} | {roi_mode} | {meta.get('snapshot_type','')}"
        preview_path = simpan_preview(
            p, depth_m, valid_mask, roi_tuple,
            label=label,
            nilai_sistem_cm=nilai_sistem_cm,
            aktual_cm=actual_cm,
            depth_roi_cm=depth_utama_cm,
            zref_cm=zref_cm,
            status=status,
        )

        hasil.append({
            "no": row.get("no", ""),
            "snapshot_npz": p,
            "snapshot": os.path.basename(p),
            "snapshot_type": meta.get("snapshot_type", ""),
            "snapshot_label": meta.get("snapshot_label", ""),
            "jenis_uji": jenis,
            "jenis_label": jenis_label,
            "nilai_aktual_cm": actual_cm,
            "nilai_sistem_cm": nilai_sistem_cm,
            "error_cm": error_cm,
            "abs_error_cm": error_abs_cm,
            "error_persen": error_pct,
            "depth_roi_median_cm": stat["depth_median_cm"],
            "depth_roi_mean_cm": stat["depth_mean_cm"],
            "depth_roi_std_cm": stat["depth_std_cm"],
            "depth_roi_p25_cm": stat["depth_p25_cm"],
            "depth_roi_p75_cm": stat["depth_p75_cm"],
            "zref_cm": zref_cm,
            "zref_source": zref_source_detail,
            "n_valid_roi": stat["n_valid"],
            "valid_roi_persen": stat["pct_valid_roi"],
            "roi_mode": roi_mode,
            "roi_x1": roi_tuple[0], "roi_y1": roi_tuple[1], "roi_x2": roi_tuple[2], "roi_y2": roi_tuple[3],
            "status": status,
            "roi_warning": roi_warning,
            "preview_roi": preview_path,
            "catatan": row.get("catatan", ""),
        })

    if not ada_nilai:
        print("\n[INFO] CSV sudah ada, tetapi nilai_aktual_cm belum diisi.")
        print(f"Buka dan isi: {TEMPLATE_CSV}")
        print("Lalu jalankan lagi. Untuk balok, jalankan juga --pilih-roi.")
        tulis_panduan()
        return

    if not hasil:
        print("\n[ERROR] Tidak ada baris evaluasi yang berhasil diproses.")
        return

    simpan_hasil_csv(hasil)
    simpan_ringkasan(hasil)
    buat_grafik(hasil)

    print("\n[OK] Evaluasi stereo vision 05B mode ukur selesai.")
    print(f"Hasil CSV      : {HASIL_CSV}")
    print(f"Ringkasan      : {RINGKASAN_TXT}")
    print(f"Preview ROI    : {PREVIEW_DIR}")


def simpan_hasil_csv(hasil):
    fieldnames = list(hasil[0].keys())
    with open(HASIL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in hasil:
            out = {}
            for k, v in row.items():
                if isinstance(v, float):
                    out[k] = "" if not np.isfinite(v) else f"{v:.6f}"
                else:
                    out[k] = v
            writer.writerow(out)


def metrik(hasil, jenis=None):
    vals = [r for r in hasil if jenis is None or r["jenis_uji"] == jenis]
    errs = np.array([r["error_cm"] for r in vals if np.isfinite(r["error_cm"])], dtype=float)
    actual = np.array([r["nilai_aktual_cm"] for r in vals if np.isfinite(r["error_cm"])], dtype=float)
    if errs.size == 0:
        return None
    mae = float(np.mean(np.abs(errs)))
    rmse = float(np.sqrt(np.mean(errs ** 2)))
    bias = float(np.mean(errs))
    mape = float(np.mean(np.abs(errs) / np.maximum(np.abs(actual), 1e-9)) * 100.0)
    return {"n": int(errs.size), "MAE_cm": mae, "RMSE_cm": rmse, "Bias_cm": bias, "MAPE_persen": mape}


def simpan_ringkasan(hasil):
    lines = []
    lines.append("RINGKASAN EVALUASI STEREO VISION 05B — MODE UKUR")
    lines.append("=" * 55)
    lines.append(f"Waktu evaluasi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Input snapshot: {SNAPSHOT_DIR}")
    lines.append(f"Z_ref file    : {ZREF_NPZ}")
    lines.append(f"Z_ref global  : {'vertical' if PAKAI_ZREF_VERTICAL else 'raw'}")
    if PAKAI_ZREF_LOKAL_DARI_ROI_LANTAI:
        lines.append("Z_ref tinggi  : ROI LANTAI_KOSONG lokal jika tersedia")
    else:
        lines.append("Z_ref tinggi  : zref.npz global")
    lines.append(f"Statistik utama depth: {STAT_UTAMA}")
    lines.append("")
    lines.append("Konsep evaluasi:")
    lines.append("1. jarak_lantai = median depth ROI lantai dibandingkan jarak aktual kamera-lantai.")
    lines.append("2. tinggi_objek = Z_ref - median depth ROI permukaan atas balok.")
    lines.append("3. Pada versi FIX ini, Z_ref untuk tinggi_objek memakai depth ROI LANTAI_KOSONG jika tersedia.")
    lines.append("4. Snapshot MUATAN_BATUBARA dilewati karena 05B hanya untuk validasi stereo dasar.")
    lines.append("")

    for jenis in [None, "jarak_lantai", "tinggi_objek"]:
        m = metrik(hasil, jenis)
        label = "Semua uji" if jenis is None else jenis
        lines.append(f"Metrik {label}:")
        if m is None:
            lines.append("  Tidak ada data valid.")
        else:
            lines.append(f"  Jumlah data : {m['n']}")
            lines.append(f"  MAE         : {m['MAE_cm']:.3f} cm")
            lines.append(f"  RMSE        : {m['RMSE_cm']:.3f} cm")
            lines.append(f"  Bias        : {m['Bias_cm']:.3f} cm")
            lines.append(f"  MAPE        : {m['MAPE_persen']:.2f}%")
        lines.append("")

    lines.append("Detail status:")
    for r in hasil:
        lines.append(
            f"- {r['snapshot']} | {r['snapshot_type']} | {r['jenis_uji']} | aktual={r['nilai_aktual_cm']:.2f} cm | "
            f"sistem={r['nilai_sistem_cm']:.2f} cm | error={r['error_cm']:.2f} cm | "
            f"depthROI={r['depth_roi_median_cm']:.2f} cm | Zref={r['zref_cm']:.2f} cm | "
            f"validROI={r['valid_roi_persen']:.1f}% | ROI={r['roi_mode']} | {r['status']}"
        )
        if r.get("roi_warning"):
            lines.append(f"  Catatan ROI: {r['roi_warning']}")

    lines.append("")
    lines.append("Interpretasi:")
    lines.append("- Untuk lantai, error kecil menunjukkan Z_ref/depth dasar sudah baik.")
    lines.append("- Untuk balok, depthROI harus lebih kecil dari Z_ref. Jika Z_ref sekitar 56 cm dan balok 5 cm, depthROI atas balok ideal sekitar 51 cm.")
    lines.append("- Jika tinggi_objek mendekati 0 cm, ROI kemungkinan masih memilih lantai atau area gerbong besar, bukan atas balok.")
    lines.append("- Jika validROI objek rendah, tambah tekstur balok, perbaiki cahaya, dan pilih ROI lebih kecil di permukaan atas balok.")
    lines.append("- Evaluasi ini dilakukan sebelum U-Net, volume, dan massa agar dasar stereo vision terbukti terlebih dahulu.")

    with open(RINGKASAN_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def buat_grafik(hasil):
    if not MATPLOTLIB_OK:
        return
    valid = [r for r in hasil if np.isfinite(r["nilai_aktual_cm"]) and np.isfinite(r["nilai_sistem_cm"])]
    if len(valid) < 2:
        return

    x = np.array([r["nilai_aktual_cm"] for r in valid], dtype=float)
    y = np.array([r["nilai_sistem_cm"] for r in valid], dtype=float)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y)
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    if hi <= lo:
        hi = lo + 1.0
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Nilai Aktual (cm)")
    plt.ylabel("Nilai Sistem (cm)")
    plt.title("Evaluasi Stereo Vision 05B Mode Ukur: Sistem vs Aktual")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "grafik_sistem_vs_aktual_05b_mode_ukur.png"), dpi=200)
    plt.close()

    errs = y - x
    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(len(errs)), errs)
    plt.axhline(0, linestyle="--")
    plt.xticks(np.arange(len(errs)), [str(i+1) for i in range(len(errs))])
    plt.xlabel("Nomor Data")
    plt.ylabel("Error (cm)")
    plt.title("Error Evaluasi Stereo Vision 05B Mode Ukur")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "grafik_error_05b_mode_ukur.png"), dpi=200)
    plt.close()

# =========================================================
# MAIN
# =========================================================
def main():
    pastikan_folder()
    print("\n" + "="*68)
    print(" SCRIPT 05B — EVALUASI STEREO VISION MODE UKUR + ZREF LOKAL LANTAI FIX")
    print("="*68)
    print("Input:")
    print(f"- Snapshot depth : {SNAPSHOT_DIR}")
    print(f"- Z_ref          : {ZREF_NPZ}")
    print("Output:")
    print(f"- Evaluasi       : {OUTPUT_DIR}")

    if "--pilih-roi" in sys.argv or "--roi" in sys.argv:
        mode_pilih_roi()
        return

    proses_evaluasi()


if __name__ == "__main__":
    main()
