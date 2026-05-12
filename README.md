# Prediksi Kematian pada Pasien COVID-19

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Oktazz/pasien-covid/blob/main/pasien_covid.ipynb)

## 📋 Daftar Isi
- [Deskripsi Proyek](#deskripsi-proyek)
- [Tujuan Proyek](#tujuan-proyek)
- [Dataset](#dataset)
- [Metodologi](#metodologi)
- [Model Machine Learning](#model-machine-learning)
- [Hasil dan Evaluasi](#hasil-dan-evaluasi)
- [Instalasi dan Penggunaan](#instalasi-dan-penggunaan)
- [Struktur Proyek](#struktur-proyek)
- [Kontribusi](#kontribusi)
- [Lisensi](#lisensi)

## 🎯 Deskripsi Proyek

Proyek ini adalah sistem prediksi kematian pada pasien COVID-19 menggunakan machine learning. Proyek ini menggunakan dataset pasien COVID-19 untuk membangun model prediktif yang dapat mengidentifikasi faktor-faktor risiko tinggi dan memprediksi kemungkinan kematian pasien berdasarkan kondisi medis dan riwayat kesehatan mereka.

Dengan menggunakan berbagai algoritma machine learning seperti Logistic Regression, Random Forest Classifier, dan Decision Tree Classifier, proyek ini bertujuan untuk membantu tenaga medis dalam membuat keputusan klinis yang lebih baik dan meningkatkan tingkat kelangsungan hidup pasien COVID-19.

## 🎓 Tujuan Proyek

1. **Identifikasi Faktor Risiko**: Mengidentifikasi faktor-faktor yang paling berpengaruh terhadap kematian pasien COVID-19
2. **Prediksi Akurat**: Membangun model prediktif yang akurat untuk memprediksi risiko kematian
3. **Optimasi Sumber Daya**: Membantu mengoptimalkan alokasi sumber daya medis berdasarkan prediksi risiko
4. **Penelitian Medis**: Memberikan wawasan berharga untuk penelitian epidemiologi dan kesehatan masyarakat

## 📊 Dataset

### Sumber Data
- **Dataset Utama**: CovidData.csv
- **Jumlah Sample**: 6,000 pasien (dari 1 juta data asli)
- **Jumlah Fitur**: 21 fitur

### Deskripsi Fitur

| Fitur | Deskripsi | Tipe |
|-------|-----------|------|
| USMER | Tipe Fasilitas Medis | Biner |
| MEDICAL_UNIT | Unit Medis | Numerik |
| SEX | Jenis Kelamin (0: Perempuan, 1: Laki-laki) | Biner |
| PATIENT_TYPE | Tipe Pasien (0: Rawat Jalan, 1: Rawat Inap) | Biner |
| AGE | Usia Pasien | Numerik |
| INTUBED | Pasien Diintubasi (0: Tidak, 1: Ya) | Biner |
| PNEUMONIA | Pneumonia (0: Tidak, 1: Ya) | Biner |
| PREGNANT | Kehamilan (0: Tidak, 1: Ya) | Biner |
| DIABETES | Diabetes (0: Tidak, 1: Ya) | Biner |
| COPD | COPD (0: Tidak, 1: Ya) | Biner |
| ASTHMA | Asma (0: Tidak, 1: Ya) | Biner |
| INMSUPR | Imunosupresi (0: Tidak, 1: Ya) | Biner |
| HIPERTENSION | Hipertensi (0: Tidak, 1: Ya) | Biner |
| OTHER_DISEASE | Penyakit Lain (0: Tidak, 1: Ya) | Biner |
| CARDIOVASCULAR | Penyakit Kardiovaskular (0: Tidak, 1: Ya) | Biner |
| OBESITY | Obesitas (0: Tidak, 1: Ya) | Biner |
| RENAL_CHRONIC | Penyakit Ginjal Kronis (0: Tidak, 1: Ya) | Biner |
| TOBACCO | Penggunaan Tembakau (0: Tidak, 1: Ya) | Biner |
| CLASIFFICATION_FINAL | Klasifikasi Final | Numerik |
| ICU | Perawatan ICU (0: Tidak, 1: Ya) | Biner |
| **DIED** | **Target: Kematian (0: Tidak, 1: Ya)** | **Biner** |

### Statistik Dataset

- **Total Sampel**: 6,000
- **Sampel Hidup (DIED=0)**: 5,570 (92.83%)
- **Sampel Meninggal (DIED=1)**: 430 (7.17%)
- **Data Duplikat**: 1,890 (dihapus)
- **Missing Value**: 0

## 🔧 Metodologi

### 1. Pengumpulan dan Pembersihan Data
```
├── Import Libraries (pandas, numpy, matplotlib, seaborn)
├── Koneksi Google Drive
├── Baca Dataset CSV
└── Sampling 6,000 dari 1 juta data
```

### 2. Eksplorasi dan Analisis Data (EDA)
- Visualisasi distribusi data
- Analisis statistik deskriptif
- Identifikasi pola dan outliers

### 3. Pengolahan Data
- **Konversi Fitur**: Mengubah DATE_DIED menjadi kolom biner DIED
- **Normalisasi Nilai**: Mengkonversi kolom dengan nilai (1,2,97,98,99) ke (0,1)
- **Penanganan Kategorikal**: 
  - SEX: 1 (Perempuan) → 0, 2 (Laki-laki) → 1
  - PREGNANT: Handling khusus untuk laki-laki
- **Hapus Data Duplikat**: Menghapus 1,890 data duplikat

### 4. Preprocessing dan Normalisasi
- **Train-Test Split**: 70% training, 30% testing
- **Normalisasi**: StandardScaler untuk normalisasi fitur numerik
- **Penanganan Imbalanced Data**: SMOTE (Synthetic Minority Over-sampling Technique)
  - Sebelum SMOTE: Kelas 0 = 3,907, Kelas 1 = 293
  - Sesudah SMOTE: Kelas 0 = 3,907, Kelas 1 = 3,907

### 5. Analisis Korelasi
Menghitung korelasi setiap fitur dengan target untuk mengidentifikasi fitur paling penting.

## 🤖 Model Machine Learning

Proyek ini mengimplementasikan tiga algoritma klasifikasi:

### 1. Logistic Regression
```python
LogisticRegression(max_iter=1000, class_weight='balanced')
```
- **Akurasi**: 89.83%
- **Precision (Kelas 1)**: 0.41
- **Recall (Kelas 1)**: 0.81
- **F1-Score**: 0.55

### 2. Random Forest Classifier
```python
RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
```
- **Akurasi**: 91.78% ✅ **Best Model**
- **Precision (Kelas 1)**: 0.47
- **Recall (Kelas 1)**: 0.65
- **F1-Score**: 0.55

### 3. Decision Tree Classifier
```python
DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=3)
```
- **Akurasi**: 88.11%
- **Precision (Kelas 1)**: 0.48
- **Recall (Kelas 1)**: 0.59
- **F1-Score**: 0.53

## 📈 Hasil dan Evaluasi

### Perbandingan Model

| Model | Akurasi | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | 89.83% | 0.41 | 0.81 | 0.55 |
| **Random Forest** | **91.78%** | **0.47** | **0.65** | **0.55** |
| Decision Tree | 88.11% | 0.48 | 0.59 | 0.53 |

### Confusion Matrix - Random Forest (Best Model)

```
Predicted    Tidak(0)    Ya(1)
Actual
Tidak(0)     1562        101
Ya(1)         48         89
```

- **True Negative (TN)**: 1,562
- **False Positive (FP)**: 101
- **False Negative (FN)**: 48
- **True Positive (TP)**: 89

### Key Findings

1. **Model Terbaik**: Random Forest Classifier dengan akurasi 91.78%
2. **Trade-off**: Ada trade-off antara precision dan recall
   - High Recall (0.81) pada Logistic Regression: Mendeteksi lebih banyak kasus positif
   - Balanced Performance pada Random Forest: Better generalization
3. **Imbalanced Data**: SMOTE efektif meningkatkan performa pada minority class

## 💻 Instalasi dan Penggunaan

### Prerequisites
- Python 3.8+
- Google Colab (untuk menjalankan notebook)
- Google Drive (untuk akses dataset)

### Library yang Digunakan

```python
# Data Processing
pandas>=1.3.0
numpy>=1.21.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Machine Learning
scikit-learn>=0.24.0
imbalanced-learn>=0.8.0

# Google Colab
google-colab
```

### Langkah-Langkah Penggunaan

1. **Buka Notebook di Google Colab**
   ```
   https://colab.research.google.com/github/Oktazz/pasien-covid/blob/main/pasien_covid.ipynb
   ```

2. **Mount Google Drive**
   - Jalankan cell untuk mount Google Drive
   - Pastikan dataset CovidData.csv ada di path yang sesuai

3. **Jalankan Setiap Cell Secara Berurutan**
   - Mulai dari import libraries hingga evaluasi model

4. **Interpretasi Hasil**
   - Lihat visualization dan metrics untuk memahami performa model

### Alternatif: Menjalankan Lokal

```bash
# Clone repository
git clone https://github.com/Oktazz/pasien-covid.git
cd pasien-covid

# Install dependencies
pip install -r requirements.txt

# Jalankan notebook
jupyter notebook pasien_covid.ipynb
```

## 📁 Struktur Proyek

```
pasien-covid/
│
├── README.md                    # Dokumentasi proyek
├── pasien_covid.ipynb          # Jupyter Notebook utama
├── requirements.txt            # Dependencies
│
└── assets/
    ├── correlation_plot.png    # Plot korelasi fitur
    ├── confusion_matrix_rf.png # Confusion matrix RF
    └── distribution_plot.png   # Distribusi target
```

## 🔍 Detail Teknis

### Tahap-Tahap Pipeline ML

```
Raw Data
    ↓
Data Cleaning (Remove Duplicates)
    ↓
Feature Engineering (Encoding, Normalization)
    ↓
Train-Test Split (70-30)
    ↓
Imbalance Handling (SMOTE)
    ↓
StandardScaler Normalization
    ↓
Model Training (LR, RF, DT)
    ↓
Model Evaluation & Comparison
    ↓
Best Model Selection
```

### Parameter Tuning

#### Random Forest (Best Model)
```python
n_estimators: 100         # Jumlah pohon
random_state: 42          # Untuk reproducibility
class_weight: 'balanced'  # Menangani imbalanced data
```

#### Logistic Regression
```python
max_iter: 1000            # Iterasi maksimal
class_weight: 'balanced'  # Penimbang untuk imbalanced data
```

#### Decision Tree
```python
max_depth: 3              # Kedalaman maksimal
random_state: 42          # Untuk reproducibility
class_weight: 'balanced'  # Penimbang untuk imbalanced data
```

## 📚 Referensi dan Teori

### Konsep yang Digunakan

1. **SMOTE (Synthetic Minority Over-sampling Technique)**
   - Teknik untuk menangani imbalanced dataset
   - Menghasilkan synthetic samples untuk minority class

2. **StandardScaler Normalization**
   - Standarisasi fitur ke mean=0, std=1
   - Penting untuk algoritma berbasis distance

3. **Class Weight Balancing**
   - Memberikan bobot lebih pada minority class
   - Mencegah bias terhadap majority class

4. **Train-Test Split**
   - 70% untuk training, 30% untuk testing
   - Mengevaluasi performa pada unseen data

## 🚀 Future Improvements

1. **Hyperparameter Tuning**: GridSearchCV atau RandomizedSearchCV untuk optimasi parameter
2. **Feature Selection**: Menggunakan SelectKBest atau RFE untuk fitur terpenting
3. **Ensemble Methods**: Voting Classifier atau Stacking untuk performa lebih baik
4. **Cross-Validation**: K-Fold Cross-Validation untuk evaluasi lebih robust
5. **Model Interpretability**: SHAP values untuk menjelaskan prediksi model
6. **Deep Learning**: Neural Networks untuk pattern recognition yang lebih complex
7. **Real-time Prediction**: API deployment untuk prediksi real-time

## ⚠️ Keterbatasan dan Pertimbangan

1. **Data Imbalance**: Dataset sangat tidak seimbang (7.17% positif), memerlukan teknik khusus
2. **Sample Size**: Hanya menggunakan 6,000 dari 1 juta data asli
3. **Temporal Data**: Model tidak mempertimbangkan temporal patterns
4. **Geographic Variation**: Tidak ada informasi lokasi geografis
5. **Data Quality**: Dataset publik mungkin memiliki quality issues
6. **Ethical Considerations**: Model hanya untuk support decision-making, bukan replacement dokter

## 👨‍💻 Author

**Oktazz**
- GitHub: [@Oktazz](https://github.com/Oktazz)
- Email: [Contact via GitHub]

## 🤝 Kontribusi

Kontribusi sangat diterima! Untuk berkontribusi:

1. Fork repository ini
2. Buat branch fitur (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buka Pull Request

## 📝 Lisensi

Proyek ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 🙏 Acknowledgments

- Dataset COVID-19 dari sumber publik
- scikit-learn team untuk tools ML yang excellent
- Google Colab untuk computational resources
- Komunitas open-source

---

**Catatan**: Model ini dibuat untuk tujuan penelitian dan edukasi. Jangan gunakan sebagai pengganti konsultasi medis profesional. Selalu konsultasikan dengan tenaga medis profesional untuk diagnosis dan treatment.

---

*Last Updated: May 2026*
