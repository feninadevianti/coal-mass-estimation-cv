<h1 align="center">Coal Mass Estimation CV</h1>

<p align="center">
  <b>Estimasi massa muatan batu bara tanpa kontak</b><br>
  Segmentasi U-Net dan stereo vision pada miniatur gerbong kereta
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-U--Net-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-Stereo%20Vision-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
</p>

<p align="center">
  <img src="hasil_19_estimasi_massa_semua_model/dashboard_visual_20260704_172156/B10_T1000_R02_gundukan_20260702_133355_dashboard_estimasi.png" width="92%" alt="Dashboard estimasi massa: citra pandangan atas, mask, peta tinggi, dan hasil akhir">
</p>

## Ringkasan

Sistem ini mengestimasi massa batu bara pada gerbong **tanpa menyentuh muatan**, sebagai alternatif timbangan rel statis atau sensor yang mahal. Sepasang foto stereo diolah menjadi tinggi tumpukan: U-Net mengenali area batu bara, stereo vision mengukur jaraknya, lalu tinggi tersebut dijumlahkan menjadi volume dan dikonversi menjadi massa. Sistem diuji pada miniatur skala laboratorium dan merupakan bagian dari skripsi Teknik Elektro, Universitas Airlangga.

## Hasil utama

| IoU segmentasi | Galat kedalaman | MAPE massa | R² massa |
|:---:|:---:|:---:|:---:|
| **0,9718** | **0,27 cm** | **4,1%** | **0,9978** |
| 30 citra uji | MAE, 7 titik uji balok | 66 sampel dan data uji | 0,9960 pada data uji |

## Cara kerja

```mermaid
flowchart LR
    A["1. Kalibrasi stereo"] --> B["2. Depth map dan Z_ref"] --> C["3. Segmentasi U-Net"] --> D["4. Height map"] --> E["5. Volume"] --> F["6. Massa"]
```

Setiap tahap memiliki kriteria kelayakan yang harus dipenuhi sebelum lanjut ke tahap berikutnya.

**[Lihat penjelasan bergambar tiap tahap](PIPELINE.md)**

## Sorotan

- **Validasi bertahap.** Setiap tahap diperiksa dengan kriteria yang jelas, misalnya RMS kalibrasi < 0,5 px dan IoU > 0,70.
- **Uji kedalaman dengan balok acuan 5 cm.** Error maksimum 0,55 cm, di bawah batas 1 cm.
- **Enam model konversi volume ke massa dibandingkan**, dievaluasi pada seluruh data dan pada data uji yang dipisahkan.

## Teknologi

Python · OpenCV (ChArUco, StereoSGBM) · PyTorch (U-Net) · NumPy · Matplotlib · Google Colab · Roboflow

<details>
<summary><b>Isi repositori</b></summary>

| Folder | Isi |
|---|---|
| `hasil_kalibrasi/` | Parameter kalibrasi, matriks stereo, dan *stereo map* |
| `hasil_evaluasi_stereo_05b_mode_ukur/` | Uji kedalaman dengan jarak lantai dan balok acuan |
| `hasil_training_unet_colab/` | Kurva training dan contoh prediksi U-Net |
| `hasil_evaluasi_unet/` | Metrik segmentasi per citra dan contoh *overlay* |
| `hasil_inference_unet/`, `hasil_integrasi_unet_depth/`, `hasil_volume_homografi/` | Contoh hasil inferensi, integrasi, dan volume (hanya beberapa contoh) |
| `hasil_evaluasi_volume_homografi/` | Evaluasi volume |
| `hasil_18_...`, `hasil_19_...` | Perbandingan model dan dashboard estimasi massa |

Hasil lengkap dapat dibuat ulang dengan menjalankan skrip terkait.

</details>

<details>
<summary><b>Daftar skrip</b></summary>

Skrip diberi nomor sesuai urutan pengerjaan. Ringkasannya juga ada di `00_README_URUTAN_ALUR_MASSA.txt`.

| No. | Fungsi |
|---|---|
| 01 | Pengambilan citra ChArUco stereo |
| 02 | Kalibrasi intrinsik kamera kiri dan kanan |
| 03 | Kalibrasi stereo |
| 04 | Rektifikasi dan pembuatan *stereo map* |
| 05 | Evaluasi kalibrasi dan rektifikasi |
| 06 | Pembuatan depth map |
| 07 | Evaluasi akurasi kedalaman (jarak lantai dan balok) |
| 08 | Akuisisi dataset U-Net |
| 09 | Panduan anotasi manual di Roboflow |
| 10 | Preprocessing hasil ekspor Roboflow |
| 11 | Normalisasi *mask* menjadi biner |
| 12 | Training U-Net (Google Colab) |
| 13 | Inferensi segmentasi |
| 14 | Evaluasi segmentasi (IoU, Dice, precision, recall) |
| 15 | Integrasi segmentasi dan kedalaman menjadi height map |
| 16 | Perhitungan volume dengan homografi |
| 17 | Evaluasi volume |
| 18 | Perbandingan model konversi volume ke massa |
| 19 | Estimasi massa dengan semua model |
| 20 | Demo *near real-time* |

</details>

## Model dan dataset

Bobot model U-Net dan dataset tidak disertakan karena ukurannya besar. Tautan unduhan: **[akan ditambahkan]**

## Menjalankan

1. Pasang kebutuhan: `pip install -r requirements.txt`
2. Jalankan skrip 01 sampai 05 untuk kalibrasi kamera dan rektifikasi.
3. Siapkan model dan dataset (unduh dari tautan di atas, atau buat sendiri melalui skrip 08 sampai 12; skrip 12 dijalankan di Google Colab).
4. Jalankan skrip 13 dan 15 sampai 16 untuk segmentasi dan volume, lalu skrip 18 dan 19 untuk estimasi massa, atau skrip 20 untuk demo.

Sesuaikan jalur (*path*) folder di dalam tiap skrip dengan lokasi berkas di komputer Anda.

## Keterbatasan

- Sistem diuji pada miniatur skala laboratorium; kondisi lapangan (debu, getaran, pencahayaan, dan ukuran gerbong sebenarnya) belum diuji.
- Hasil di repositori ini menggunakan *baseline* 10 cm. Konfigurasi 15 cm baru diuji pada tahap kalibrasi, sehingga pengaruh *baseline* terhadap akurasi belum dapat dibandingkan.
- Evaluasi pada data uji memakai satu pembagian 80:20 (13 sampel); validasi silang berulang masih diperlukan.

---

<p align="center"><sub>Enina Devianti Farid · Teknik Elektro, Universitas Airlangga</sub></p>
