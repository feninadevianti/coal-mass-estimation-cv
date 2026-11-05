
import os
import csv
from datetime import datetime

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KALIBRASI_DIR = os.path.join(BASE_DIR, "hasil_kalibrasi")
STEREOMAP_XML = os.path.join(KALIBRASI_DIR, "stereomap.xml")
OUTPUT_DIR = os.path.join(BASE_DIR, "hasil_evaluasi_kalibrasi_stereo")

BASELINE_FISIK_CM = 10.0

RMS_MAX_PX = 1.0
BASELINE_ERROR_MAX_PCT = 5.0

def _node_empty(node):
    try:
        return node.empty()
    except Exception:
        return True


def baca_matrix(fs, nama_node):
    node = fs.getNode(nama_node)
    if _node_empty(node):
        return None
    mat = node.mat()
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

def load_stereomap_info():
    if not os.path.exists(STEREOMAP_XML):
        raise FileNotFoundError(f"stereomap.xml tidak ditemukan: {STEREOMAP_XML}")

    fs = cv2.FileStorage(STEREOMAP_XML, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Gagal membuka XML: {STEREOMAP_XML}")

    nodes = {}
    for nama in ["stereoMapL_x", "stereoMapL_y", "stereoMapR_x", "stereoMapR_y", "Q", "R", "T", "P1", "P2"]:
        nodes[nama] = baca_matrix(fs, nama)

    image_width = baca_int(fs, "image_width", default=None)
    image_height = baca_int(fs, "image_height", default=None)
    rms_stereo = baca_float(fs, "rms_stereo", default=None)
    baseline_mm = baca_float(fs, "baseline_mm", default=None)

    fs.release()

    # fallback: baseline dari T jika baseline_mm tidak ada
    if baseline_mm is None and nodes.get("T") is not None:
        T = np.asarray(nodes["T"], dtype=float).reshape(-1)
        baseline_mm = float(np.linalg.norm(T) * 1000.0) if np.linalg.norm(T) < 10 else float(np.linalg.norm(T))

    return {
        "nodes": nodes,
        "image_width": image_width,
        "image_height": image_height,
        "rms_stereo": rms_stereo,
        "baseline_mm": baseline_mm,
    }

def status_bool(cond):
    return "LULUS" if cond else "TIDAK LULUS"


def evaluasi(info):
    nodes = info["nodes"]

    wajib_nodes = ["stereoMapL_x", "stereoMapL_y", "stereoMapR_x", "stereoMapR_y", "Q"]
    node_status = {nama: nodes.get(nama) is not None for nama in wajib_nodes}
    semua_node_wajib_ada = all(node_status.values())

    rms = info["rms_stereo"]
    if rms is None:
        rms_lulus = False
    else:
        rms_lulus = rms <= RMS_MAX_PX

    baseline_cm = None
    baseline_error_pct = None
    baseline_lulus = False
    if info["baseline_mm"] is not None:
        baseline_cm = float(info["baseline_mm"]) / 10.0
        baseline_error_pct = abs(baseline_cm - BASELINE_FISIK_CM) / BASELINE_FISIK_CM * 100.0
        baseline_lulus = baseline_error_pct <= BASELINE_ERROR_MAX_PCT

    hasil_akhir = semua_node_wajib_ada and rms_lulus and baseline_lulus

    return {
        "semua_node_wajib_ada": semua_node_wajib_ada,
        "node_status": node_status,
        "rms_stereo_px": rms,
        "rms_lulus": rms_lulus,
        "baseline_cm": baseline_cm,
        "baseline_fisik_cm": BASELINE_FISIK_CM,
        "baseline_error_pct": baseline_error_pct,
        "baseline_lulus": baseline_lulus,
        "image_width": info["image_width"],
        "image_height": info["image_height"],
        "hasil_akhir": hasil_akhir,
    }

def simpan_hasil(hasil):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"evaluasi_kalibrasi_stereo_{waktu}.csv")
    txt_path = os.path.join(OUTPUT_DIR, f"ringkasan_evaluasi_kalibrasi_stereo_{waktu}.txt")

    rows = [
        ["stereomap_node_wajib", "ada", hasil["semua_node_wajib_ada"], status_bool(hasil["semua_node_wajib_ada"])],
        ["rms_stereo_px", f"<= {RMS_MAX_PX}", hasil["rms_stereo_px"], status_bool(hasil["rms_lulus"])],
        ["baseline_error_pct", f"<= {BASELINE_ERROR_MAX_PCT}", hasil["baseline_error_pct"], status_bool(hasil["baseline_lulus"])],
        ["hasil_akhir", "semua kriteria lulus", hasil["hasil_akhir"], status_bool(hasil["hasil_akhir"])],
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "kriteria", "nilai", "status"])
        writer.writerows(rows)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("EVALUASI KALIBRASI STEREO DAN REKTIFIKASI\n")
        f.write("=========================================\n\n")
        f.write(f"File stereomap : {STEREOMAP_XML}\n")
        f.write(f"Ukuran image   : {hasil['image_width']} x {hasil['image_height']}\n\n")
        f.write("Node wajib:\n")
        for nama, ada in hasil["node_status"].items():
            f.write(f"- {nama}: {'ADA' if ada else 'TIDAK ADA'}\n")
        f.write("\n")
        f.write(f"RMS stereo          : {hasil['rms_stereo_px']} px | kriteria <= {RMS_MAX_PX} px | {status_bool(hasil['rms_lulus'])}\n")
        f.write(f"Baseline fisik      : {BASELINE_FISIK_CM:.3f} cm\n")
        f.write(f"Baseline kalibrasi  : {hasil['baseline_cm']} cm\n")
        f.write(f"Error baseline      : {hasil['baseline_error_pct']} % | kriteria <= {BASELINE_ERROR_MAX_PCT}% | {status_bool(hasil['baseline_lulus'])}\n")
        f.write("\n")
        f.write(f"HASIL AKHIR         : {status_bool(hasil['hasil_akhir'])}\n")
        f.write("\nCatatan: rectification tetap perlu dicek visual menggunakan Script 04.\n")

    print("\n[OK] Evaluasi kalibrasi stereo selesai")
    print(f"CSV : {csv_path}")
    print(f"TXT : {txt_path}")

def main():
    print("\n" + "=" * 72)
    print("SCRIPT 15C — EVALUASI KALIBRASI STEREO DAN REKTIFIKASI")
    print("=" * 72)
    print(f"Stereomap : {STEREOMAP_XML}")
    print(f"Output    : {OUTPUT_DIR}")
    print("=" * 72)

    info = load_stereomap_info()
    hasil = evaluasi(info)
    simpan_hasil(hasil)


if __name__ == "__main__":
    main()
