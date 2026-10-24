# Panduan Mengerjakan & Memasarkan Project Ini (Bahasa Indonesia)

Dokumen ini untuk kamu, bukan untuk recruiter. README utama berbahasa Inggris karena itu standar di GitHub.

## A. Langkah kerja (perkiraan 1-2 minggu santai)

1. **Buat repo GitHub** bernama `defect-detection-mlops`, push semua isi folder ini.
2. **Siapkan dataset NEU-DET** (gambar + anotasi kotak berformat `.xml`). Ini beda dari NEU-CLS yang hanya punya label per gambar. Cari "NEU surface defect database" atau "NEU-DET" di Kaggle; pakai mirror mana pun asal ada file `.xml`-nya.
3. **Buka `notebooks/colab_train.ipynb` di Google Colab** (runtime: T4 GPU). Edit `REPO_URL`, lalu jalankan sel dari atas ke bawah.
   - Butuh `kaggle.json` (Kaggle -> Settings -> API -> Create New Token).
   - Slug dataset di notebook adalah tebakan terbaikku dan belum kuverifikasi. Sel pengecekan akan memberi tahu kalau file `.xml` tidak ditemukan.
   - Training pertama akan mengunduh bobot `yolo26n.pt` (COCO) otomatis, jadi Colab harus punya internet.
4. **Download `artifacts.zip`** di akhir notebook, ekstrak ke repo lokal (path yang sama), lalu commit:
   `models/model.onnx`, `models/classes.json`, `metrics/`, `reports/`, `examples/`.
5. **Isi tabel Results di README** dengan angka asli dari `metrics/eval.json` (jangan dikarang; recruiter sering bertanya detailnya).
6. **Jalankan lokal**: `make install`, `make serve`, `make docker-build`. Pastikan API dan Docker jalan di laptopmu.
7. **Aktifkan CI**: push ke `main`, cek tab *Actions* hijau. Kalau gate gagal karena mAP@50 di bawah 0.65, sesuaikan angkanya dengan hasil asli (di `ci.yml`), jangan asal diturunkan tanpa alasan.
8. **Deploy demo**: buat akun Hugging Face, buat *write token*. Di GitHub repo -> Settings -> Secrets: `HF_TOKEN`; Variables: `HF_SPACE_ID` (mis. `username/steel-defect-demo`). Lalu `git tag v1.0.0 && git push --tags`. Workflow CD akan build image, smoke test, push ke GHCR, dan deploy Space.
9. **Tempel link demo** di README, bagian paling atas.
10. **(Opsional, nilai plus)** Setup DVC remote gratis (DagsHub atau Google Drive) supaya `dvc pull` bisa dipakai orang lain.

## B. Hal yang perlu kamu eksperimenkan sendiri (ini yang bikin project terasa milikmu)

- **Jangan terlalu serius dengan selisih kecil antar run.** Test set-mu cuma 270 gambar (607 kotak). Dua run dengan resep yang sama bisa beda sekitar 1 poin mAP, dan AP per kelas bisa bergeser beberapa poin. Bandingkan model dengan skor **validasi**, dan kalau mau tahu seberapa besar noise-nya, jalankan varian yang hanya beda `seed`.
- **Jalankan sweep** (`make sweep` atau sel 4b di notebook): membandingkan beberapa varian di set *validasi* lalu otomatis memilih pemenang. Setelah itu jalankan `evaluate` dan `export` sekali untuk angka test resmi. Jangan pilih model berdasarkan skor test, itu bocor (data leakage).

- Bandingkan `yolo26n.pt` vs `yolo26s.pt` (ubah `train.model` di `params.yaml`) dan `imgsz` 320 vs 640; tulis tabel mAP, ukuran ONNX, dan latency.
- Lihat `reports/detection_report.txt` dan confusion matrix: kelas mana yang AP-nya paling rendah? Buka `reports/predictions_grid.png` dan tulis analisis 1 paragraf (misalnya *patches* dan *rolled-in scale* sering sulit karena batasnya kabur).
- Jalankan `make drift-demo` dan screenshot hasilnya.
- Pelajari kenapa `nms=False` dipakai di mana-mana (lihat poin di bawah dan bagian *Design decisions* README).

### Strategi menuju ~80% mAP@50 (tanpa mengakali test set)

**Cara termudah:** di Colab jalankan sel "AUTOPILOT" (atau `make best`). Urutannya: sweep varian murah, pemenang dipilih
di set **validasi**, dilatih ulang di train+val (sekitar 18% lebih banyak data), lalu dievaluasi di test **sekali**, diekspor, dan dibuat gambarnya.

Cara membaca hasilnya:
1. Lihat tabel sweep. Selisih `n_640` vs `n_640_seed2` adalah ukuran noise. Varian baru hanya dipercaya kalau lebih baik **dari selisih itu**.
2. Kalau resolusi membantu, tambahkan `n_1280` lalu `s_960` (`make best SWEEP="n_640 n_960 n_1280"`). `l_960` terakhir (paling mahal, bukti manfaatnya paling lemah).
3. Angka test dari run autopilot itulah yang kamu laporkan. Jangan mengulang sampai test mencapai 80: itu mengakali test set.
4. Untuk klaim yang lebih kuat, latih pemenang dengan 3 seed dan tulis rata-rata serta rentangnya.
5. `models/best_sweep_winner.pt` menyimpan model sebelum refit, kalau kamu mau membandingkan.

## C. Poin CV / LinkedIn (isi angka setelah training)

- Built an end-to-end steel-defect **detection** system (YOLO26, transfer learning from COCO), reaching **XX mAP@50** on a held-out test set.
- Implemented MLOps stack: DVC pipelines, MLflow experiment tracking, NMS-free ONNX export with parity checks, FastAPI + Docker (image **XX MB**, **XX ms** CPU latency).
- Eliminated train/serve skew by reimplementing the YOLO letterbox preprocessing for serving and unit-testing it pixel-for-pixel against Ultralytics.
- Set up CI/CD with GitHub Actions (lint, tests, model quality gate, container smoke test, auto-deploy to Hugging Face Spaces) and drift monitoring (KS tests on image statistics and detection outputs).

## D. Pertanyaan wawancara yang mungkin muncul

| Pertanyaan | Poin jawaban |
|---|---|
| Kenapa YOLO, bukan klasifikasi biasa? | Satu strip bisa punya banyak cacat dan lokasinya penting; deteksi memberi kelas + kotak. YOLO tetap CNN. |
| Apa itu transfer learning di sini? | Mulai dari bobot COCO (`yolo26n.pt`), fine-tune di NEU-DET yang cuma 1.800 gambar. |
| Apa itu mAP@50 vs mAP@50-95? | AP rata-rata antar kelas pada ambang IoU 0.5, vs rata-rata ambang 0.5 sampai 0.95 (lebih ketat soal ketepatan kotak). |
| Apa bedanya NMS-free? | Head YOLO26 langsung mengeluarkan deteksi final tanpa Non-Max Suppression, jadi serving lebih sederhana dan latensinya lebih stabil. |
| Kenapa `nms=False` dipakai di train, evaluasi, dan export? | Supaya mAP yang dilaporkan berasal dari model yang sama dengan yang di-deploy. Default Ultralytics memakai head lain (dengan NMS), sehingga angkanya bisa beda dari model ONNX. |
| Apa itu train/serve skew dan bagaimana kamu mengujinya? | Preprocessing training dan serving berbeda. Resize PIL ternyata menghasilkan deteksi berbeda dari OpenCV; sekarang ada test yang membandingkan letterbox dengan milik Ultralytics piksel demi piksel. |
| Bagaimana kamu mencegah data leakage? | Dedup MD5 sebelum split, split stratified, test set tidak dipakai memilih model (seleksi pakai set validasi). |
| Apa yang terjadi kalau kondisi pabrik berubah? | Drift monitor (KS test) memberi sinyal; langkah berikutnya retraining otomatis. |
| Apa keterbatasan project ini? | Dataset bersih dan seragam, ada gap ke kondisi pabrik nyata; lisensi AGPL Ultralytics untuk penggunaan komersial. |

## E. Jujur soal batasan

- NEU-DET relatif mudah dibanding data pabrik nyata. Katakan itu sendiri di wawancara; recruiter yang berpengalaman menghargai kejujuran ini.
- Ultralytics berlisensi **AGPL-3.0**: aman untuk portofolio open-source, tapi untuk produk komersial tertutup perlu lisensi enterprise. Menyebut ini di wawancara menunjukkan kamu paham sisi bisnisnya.
- Cek juga lisensi dataset sebelum mengunggah gambarnya ke repo publik (repo ini hanya menyimpan sedikit contoh di `examples/`).
- Semua pengujian di repo ini memakai data sintetis. Angka mAP asli baru ada setelah kamu melatih di NEU-DET sungguhan.
