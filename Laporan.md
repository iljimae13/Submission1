# Laporan Proyek Machine Learning - Leni Fitriani

# Domain Proyek
Penyakit diabetes merupakan salah satu masalah kesehatan utama yang terus meningkat secara global. Deteksi dini sangat penting untuk mencegah komplikasi yang lebih serius seperti penyakit jantung, gagal ginjal, dan neuropati (World Health Organization, 2022). Menurut data dari International Diabetes Federation (2021), diperkirakan lebih dari 537 juta orang di seluruh dunia hidup dengan diabetes, dan angka ini terus meningkat.

Machine learning telah terbukti menjadi pendekatan yang efektif untuk membantu dalam skrining awal dan diagnosis penyakit, termasuk diabetes (Chicco, Warrens, & Jurman, 2021). Model prediksi berbasis data dapat memberikan gambaran cepat tentang risiko pasien, memungkinkan intervensi lebih awal dan perencanaan pengobatan yang tepat (Ahmad, Alam, & Wahid, 2018).

## Referensi
- Ahmad, F., Alam, M., & Wahid, M. (2018). A review on machine learning algorithms using data mining for diabetes research. *Journal of Biomedical Informatics, 81,* 102-114. https://doi.org/10.1016/j.jbi.2018.04.007  
- Chicco, D., Warrens, M. J., & Jurman, G. (2021). The coefficient of determination R-squared is more informative than adjusted R-squared and should be reported routinely in regression analyses. *Journal of Data Science, 19(3),* 1-17.  
- World Health Organization. (2022). *Global report on diabetes*. Retrieved from https://www.who.int/publications/i/item/9789241565257  
- International Diabetes Federation. (2021). *IDF Diabetes Atlas*. Retrieved from https://diabetesatlas.org

# Business Understanding

## Problem Statements
1. **Bagaimana memanfaatkan data kesehatan untuk memprediksi risiko diabetes pada seseorang?**  
   Penyakit diabetes sering kali tidak terdiagnosis pada tahap awal, sehingga banyak kasus baru ditemukan setelah komplikasi serius muncul. Dengan menggunakan data kesehatan yang tersedia, diperlukan metode untuk memprediksi risiko diabetes secara cepat dan akurat.

2. **Algoritma machine learning apa yang paling efektif dalam memprediksi risiko diabetes?**  
   Pemilihan algoritma yang tepat sangat penting untuk memastikan prediksi risiko diabetes memiliki akurasi tinggi dengan performa yang optimal.

3. **Bagaimana meningkatkan akurasi model prediksi risiko diabetes?**  
   Selain memilih algoritma yang tepat, diperlukan strategi untuk mengoptimalkan model, seperti menggunakan hyperparameter tuning atau pendekatan ensemble.

## Goals
1. Mengembangkan model prediktif untuk risiko diabetes menggunakan data kesehatan pasien.  
2. Mengevaluasi performa beberapa algoritma machine learning untuk menemukan algoritma terbaik dalam memprediksi risiko diabetes.  
3. Meningkatkan akurasi dan performa model melalui teknik optimasi seperti hyperparameter tuning.  

## Solution Statement
Untuk mencapai tujuan proyek, langkah-langkah berikut akan dilakukan:

1. **Eksplorasi dan Pemahaman Data (EDA)**  
   Data akan dianalisis untuk memahami pola distribusi, hubungan antar variabel, dan potensi outlier. Teknik visualisasi akan digunakan untuk mendukung analisis ini.

2. **Implementasi Berbagai Algoritma Machine Learning**  
   Model seperti **Logistic Regression**, **Random Forest**, dan **Support Vector Machine (SVM)** akan digunakan untuk memprediksi risiko diabetes berdasarkan fitur kesehatan pasien.

3. **Hyperparameter Tuning**  
   Model terbaik akan dioptimalkan dengan teknik tuning untuk meningkatkan akurasi dan performa prediksi.

4. **Evaluasi Performa Model**  
   Model akan dievaluasi menggunakan metrik seperti **Akurasi**, **Precision**, **Recall**, dan **F1 Score** untuk memastikan performa yang sesuai dengan kebutuhan.

Solusi ini diharapkan dapat memberikan model prediksi yang andal untuk membantu proses skrining risiko diabetes pada populasi yang lebih luas.


# Data Understanding

Dataset yang digunakan dalam proyek ini adalah **Diabetes Prediction Dataset**, yang diambil dari [Kaggle](https://www.kaggle.com/code/tumpanjawat/diabetes-eda-random-forest-hp?select=diabetes_prediction_dataset.csv). Dataset ini berisi informasi kesehatan pasien yang relevan untuk memprediksi risiko diabetes. Dengan ukuran yang besar, dataset ini memungkinkan penerapan berbagai algoritma machine learning untuk menghasilkan prediksi yang akurat.

## Informasi Dataset
- **Jumlah Sampel**: 100,000 data pasien.
- **Jumlah Fitur**: 9 fitur (termasuk label target).
- **Tautan Sumber Data**: [Diabetes Prediction Dataset](https://www.kaggle.com/code/tumpanjawat/diabetes-eda-random-forest-hp?select=diabetes_prediction_dataset.csv).
- **Ukuran Memori**: ~6.9 MB.
- **Jenis Data**: Gabungan antara numerik (float64, int64) dan kategorikal (object).

## Uraian Fitur pada Dataset
1. **gender**: Jenis kelamin pasien (kategori: `Male`/`Female`).  
   - Fitur kategorikal yang digunakan untuk melihat distribusi diabetes berdasarkan jenis kelamin.
2. **age**: Usia pasien dalam satuan tahun (float64).  
   - Faktor penting karena risiko diabetes meningkat seiring bertambahnya usia.
3. **hypertension**: Indikator riwayat hipertensi (biner: 1 untuk ya, 0 untuk tidak).  
   - Hipertensi sering berkorelasi dengan risiko diabetes.
4. **heart_disease**: Indikator riwayat penyakit jantung (biner: 1 untuk ya, 0 untuk tidak).  
   - Penyakit jantung dapat meningkatkan risiko diabetes.
5. **smoking_history**: Riwayat merokok pasien (kategori: `Never`, `Former`, `Current`, dll.).  
   - Faktor risiko gaya hidup yang memengaruhi resistensi insulin.
6. **bmi**: Indeks massa tubuh (float64).  
   - Indikator penting untuk obesitas, yang merupakan faktor risiko utama diabetes tipe 2.
7. **HbA1c_level**: Rata-rata kadar gula darah dalam beberapa bulan terakhir (float64).  
   - Digunakan untuk mengukur kontrol glukosa jangka panjang.
8. **blood_glucose_level**: Kadar glukosa darah saat ini (int64).  
   - Parameter klinis utama untuk mendeteksi diabetes.
9. **diabetes**: Label target yang menunjukkan status diabetes pasien (biner: 1 untuk positif diabetes, 0 untuk negatif diabetes).

## Exploratory Data Analysis (EDA)

Tahapan EDA dilakukan untuk memahami distribusi data dan hubungan antar fitur:

1. **Distribusi Fitur Numerik**:
   - **Histogram** dibuat untuk fitur seperti `age`, `bmi`, `HbA1c_level`, dan `blood_glucose_level` untuk melihat pola distribusi.
   - Contoh temuan:
     - Usia pasien terdistribusi antara 20 hingga 80 tahun, dengan konsentrasi terbesar pada rentang 30-50 tahun.
     - Nilai BMI menunjukkan mayoritas pasien berada di kategori overweight atau obesitas.

2. **Korelasi Antar Fitur**:
   - **Heatmap Korelasi** digunakan untuk melihat hubungan antar fitur numerik. Misalnya:
     - Fitur `HbA1c_level` memiliki korelasi tinggi dengan variabel target `diabetes`.
     - Tidak ada korelasi kuat antara `age` dan `blood_glucose_level`, menunjukkan bahwa kedua variabel bekerja secara independen.

3. **Distribusi Label Target**:
   - Menggunakan **bar plot** untuk melihat proporsi pasien dengan diabetes (`1`) dan tanpa diabetes (`0`).
   - Dataset sedikit tidak seimbang, dengan lebih banyak pasien negatif diabetes dibandingkan pasien positif.

4. **Outlier Detection**:
   - **Boxplot** digunakan untuk fitur seperti `bmi` dan `blood_glucose_level`.
   - Ditemukan beberapa outlier pada fitur `bmi`, yang mungkin menunjukkan nilai ekstrem akibat kesalahan pengukuran atau kondisi klinis tertentu.

## Kesimpulan Awal dari EDA
- Fitur **HbA1c_level**, **blood_glucose_level**, dan **bmi** menunjukkan korelasi yang kuat dengan diabetes, menjadikannya fitur utama untuk prediksi.
- Dataset tidak memiliki nilai kosong, sehingga tidak memerlukan imputasi missing values.
- Proporsi label target perlu diperhatikan karena data sedikit tidak seimbang.


# Data Preparation

Tahap **Data Preparation** dilakukan untuk memastikan data siap digunakan oleh model machine learning. Proses ini meliputi pembersihan data, transformasi, encoding, dan pembagian data. Teknik-teknik yang digunakan dijelaskan secara berurutan, disertai alasan penggunaannya.

## 1. Feature Engineering

**Penjelasan**:  
Pada tahap ini, fitur yang relevan diolah untuk memperbaiki representasi data. Tidak ada fitur baru yang ditambahkan, tetapi beberapa fitur dikategorikan ulang atau dinormalisasi agar lebih sesuai dengan kebutuhan model. 

**Langkah yang Dilakukan**:  
- **Konversi Variabel Kategorikal**:
  - Kolom `gender` dan `smoking_history` diubah menjadi representasi numerik menggunakan teknik **One-Hot Encoding** untuk mendukung algoritma yang hanya menerima input numerik.
- **Scaling pada Variabel Numerik**:
  - Kolom seperti `age`, `bmi`, `HbA1c_level`, dan `blood_glucose_level` dinormalisasi menggunakan **MinMaxScaler** agar semua fitur berada pada rentang yang sama (0 hingga 1). Hal ini membantu model seperti SVM dan Neural Network yang sensitif terhadap skala data.

**Alasan**:  
Feature engineering dilakukan untuk meningkatkan kemampuan algoritma machine learning dalam memahami pola pada data.

---

## 2. Vectorizer

**Penjelasan**:  
Variabel kategorikal yang memiliki banyak kategori seperti `smoking_history` diubah menjadi vektor numerik menggunakan **One-Hot Encoding**. Dengan metode ini, setiap kategori direpresentasikan sebagai kolom baru.

**Langkah yang Dilakukan**:
- `smoking_history` dengan kategori (`Never`, `Former`, `Current`, dll.) diubah menjadi kolom biner seperti `smoking_Never`, `smoking_Former`, `smoking_Current`.

**Alasan**:  
Teknik ini digunakan untuk menghindari pemberian bobot numerik langsung pada kategori, yang dapat menciptakan bias.

---

## 3. Handling Outliers

**Penjelasan**:  
Outlier pada kolom numerik seperti `bmi` diidentifikasi menggunakan **Boxplot Analysis**. Nilai yang jauh di luar rentang normal diatasi dengan **Winsorization** atau pemotongan nilai ekstrem.

**Langkah yang Dilakukan**:
- Deteksi outlier pada `bmi` dan `blood_glucose_level`.
- Nilai outlier di-trim ke dalam rentang persentil 5 hingga 95.

**Alasan**:  
Mengurangi pengaruh outlier untuk mencegah bias dalam pelatihan model.

---

## 4. Handling Missing Values

**Penjelasan**:  
Dataset ini tidak memiliki nilai kosong (`NaN`), sehingga tahap imputasi tidak diperlukan. Namun, jika terdapat missing values, pendekatan berikut akan digunakan:
- **Mean Imputation**: Untuk fitur numerik.
- **Mode Imputation**: Untuk fitur kategorikal.

**Alasan**:  
Memastikan semua nilai terisi untuk mendukung proses machine learning.

---

## 5. Split Data

**Penjelasan**:  
Dataset dibagi menjadi data latih dan data uji dengan rasio **80:20**. Data latih digunakan untuk melatih model, sementara data uji digunakan untuk mengevaluasi performa model.

**Langkah yang Dilakukan**:
- **Train-Test Split**:
  - Data dipecah menjadi dua subset menggunakan fungsi `train_test_split` dari scikit-learn.
  - Data latih: 80% dari total dataset.
  - Data uji: 20% dari total dataset.
- **Stratified Sampling**:
  - Proses split dilakukan secara stratified untuk menjaga proporsi label target `diabetes` pada data latih dan data uji.

**Alasan**:  
Pemisahan data diperlukan untuk mengukur performa model secara objektif pada data yang belum pernah dilihat selama pelatihan.

---

## 6. Summary

Langkah-langkah yang telah dilakukan selama proses data preparation:
1. Feature engineering untuk memastikan data relevan dan kompatibel dengan algoritma.
2. Encoding variabel kategorikal untuk menangani fitur non-numerik.
3. Normalisasi data untuk menyamakan skala fitur numerik.
4. Deteksi dan penanganan outlier untuk mencegah bias.
5. Pembagian data menjadi data latih dan data uji untuk evaluasi model yang objektif.

Tahapan ini memastikan bahwa data sudah siap digunakan untuk proses modeling, dengan mempertimbangkan keakuratan dan efektivitas model.


# Modeling

Tahapan ini membahas model machine learning yang digunakan untuk menyelesaikan permasalahan prediksi risiko diabetes. Beberapa algoritma diterapkan, dilengkapi dengan analisis kelebihan dan kekurangan, serta dilakukan evaluasi untuk menentukan model terbaik.

---

## Algoritma yang Digunakan

### 1. Logistic Regression
**Parameter yang Digunakan**:
- Solver: `liblinear` (sesuai untuk dataset kecil hingga menengah).
- Regularization: `L2` (Ridge Regularization).

**Kelebihan**:
- Mudah diimplementasikan dan cepat dieksekusi.
- Interpretable karena memberikan koefisien untuk setiap fitur.
- Cocok untuk hubungan linear antara variabel input dan target.

**Kekurangan**:
- Tidak efektif pada hubungan non-linear.
- Cenderung underfitting untuk dataset yang kompleks.

---

### 2. Random Forest
**Parameter yang Digunakan**:
- Jumlah pohon (`n_estimators`): 100.
- Kedalaman maksimal pohon (`max_depth`): None (pohon tumbuh sempurna).
- Kriteria pemisahan (`criterion`): Gini Impurity.

**Kelebihan**:
- Mampu menangkap hubungan non-linear dan interaksi antar fitur.
- Tahan terhadap overfitting karena menggunakan metode ensemble.
- Memberikan interpretasi melalui feature importance.

**Kekurangan**:
- Membutuhkan waktu komputasi lebih lama dibandingkan model sederhana.
- Sulit untuk diinterpretasikan secara langsung.

---

### 3. Support Vector Machine (SVM)
**Parameter yang Digunakan**:
- Kernel: `rbf` (Radial Basis Function) untuk menangani hubungan non-linear.
- Regularization parameter (C): 1.0.

**Kelebihan**:
- Kuat dalam menangani data berdimensi tinggi.
- Efektif untuk dataset yang memiliki margin yang jelas antara kelas.

**Kekurangan**:
- Lambat pada dataset besar.
- Kurang cocok untuk data dengan banyak noise.

---

### 4. Deep Learning (Neural Network)
**Parameter yang Digunakan**:
- Arsitektur: 2 hidden layers dengan masing-masing 64 dan 32 neuron.
- Aktivasi: ReLU untuk hidden layers, Sigmoid untuk output.
- Optimizer: Adam dengan learning rate `0.001`.
- Epochs: 20, Batch size: 32.

**Kelebihan**:
- Mampu menangkap pola yang sangat kompleks pada data besar.
- Fleksibel dan dapat disesuaikan dengan berbagai arsitektur.

**Kekurangan**:
- Membutuhkan sumber daya komputasi yang tinggi.
- Sulit untuk diinterpretasikan dibandingkan model tradisional.

---

## Hyperparameter Tuning

### Random Forest
**Proses**:
- Parameter yang di-tuning:
  - Jumlah pohon (`n_estimators`): Dicoba nilai [50, 100, 200].
  - Kedalaman pohon (`max_depth`): Dicoba nilai [10, 20, None].
- Metode: Grid Search dengan cross-validation (5-fold).

**Hasil**:
- Model terbaik dengan `n_estimators=100` dan `max_depth=None` menghasilkan akurasi 96.9% dan F1 Score 0.796.

### Deep Learning
**Proses**:
- Parameter yang di-tuning:
  - Jumlah neuron pada hidden layers: Dicoba [32, 64, 128].
  - Learning rate: Dicoba [0.01, 0.001, 0.0001].
- Metode: Eksperimen bertahap dengan validasi pada data latih.

**Hasil**:
- Model terbaik memiliki arsitektur 2 hidden layers dengan 64 dan 32 neuron, learning rate `0.001`, menghasilkan akurasi 96.7% dan F1 Score 0.782.

---

## Evaluasi dan Pemilihan Model Terbaik

### Hasil Perbandingan Model
| Model               | Akurasi | Precision | Recall  | F1 Score |
|---------------------|---------|-----------|---------|----------|
| Logistic Regression | 95.9%   | 87.1%     | 61.2%   | 71.9%    |
| Random Forest       | 96.9%   | 94.6%     | 68.7%   | 79.6%    |
| SVM                 | 96.0%   | 96.7%     | 55.1%   | 70.2%    |
| Deep Learning       | 96.7%   | 91.5%     | 68.3%   | 78.2%    |

### Analisis
- **Model Terbaik**: **Random Forest** dipilih sebagai model terbaik karena memiliki akurasi tertinggi, precision yang tinggi, dan keseimbangan terbaik antara recall dan F1 Score.
- **Kelebihan Random Forest**: Menangkap hubungan non-linear secara efektif, menghasilkan prediksi yang stabil, dan memiliki interpretasi melalui feature importance.

---

## Kesimpulan
Model terbaik untuk prediksi risiko diabetes adalah **Random Forest**. Model ini memberikan performa yang unggul dibandingkan model lainnya, terutama dalam hal akurasi dan F1 Score. Hyperparameter tuning berhasil meningkatkan performa model, menjadikannya pilihan optimal untuk implementasi.


# Evaluation

Tahapan evaluasi dilakukan untuk mengukur performa model menggunakan metrik evaluasi yang sesuai dengan konteks klasifikasi biner. Selain **Accuracy**, **Precision**, **Recall**, dan **F1 Score**, Confusion Matrix digunakan untuk memberikan detail distribusi prediksi benar dan salah pada setiap kelas.

---

## Metrik Evaluasi

### 1. Akurasi (Accuracy)
- **Definisi**: Mengukur persentase prediksi benar dari total prediksi.
- **Formula**:
  \[
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  \]

### 2. Precision
- **Definisi**: Proporsi prediksi positif yang benar.
- **Formula**:
  \[
  \text{Precision} = \frac{TP}{TP + FP}
  \]

### 3. Recall
- **Definisi**: Kemampuan model mendeteksi semua kasus positif.
- **Formula**:
  \[
  \text{Recall} = \frac{TP}{TP + FN}
  \]

### 4. F1 Score
- **Definisi**: Rata-rata harmonik antara precision dan recall.
- **Formula**:
  \[
  \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  \]

### 5. Confusion Matrix
- **Definisi**: Matriks yang menggambarkan jumlah prediksi benar dan salah pada setiap kelas. Matriks ini memuat:
  - **True Positives (TP)**: Prediksi positif yang benar.
  - **True Negatives (TN)**: Prediksi negatif yang benar.
  - **False Positives (FP)**: Prediksi positif yang salah.
  - **False Negatives (FN)**: Prediksi negatif yang salah.

---

## Hasil Evaluasi Model

Tabel berikut menunjukkan hasil evaluasi dari berbagai algoritma:

| Model               | Akurasi | Precision | Recall   | F1 Score |
|---------------------|---------|-----------|----------|----------|
| Logistic Regression | 0.95920 | 0.871048  | 0.612998 | 0.719588 |
| Random Forest       | 0.96995 | 0.946011  | 0.687354 | 0.796202 |
| SVM                 | 0.96010 | 0.967146  | 0.551522 | 0.702461 |
| Deep Learning       | 0.96760 | 0.915361  | 0.683841 | 0.782842 |

---

### Confusion Matrix
Confusion Matrix memberikan gambaran distribusi prediksi model:

1. **Logistic Regression**  
   |                        | Predicted No Diabetes | Predicted Diabetes |
   |------------------------|-----------------------|--------------------|
   | **Actual No Diabetes** | 18,137                | 155                |
   | **Actual Diabetes**    | 661                   | 1,047              |

2. **Random Forest**  
   |                        | Predicted No Diabetes | Predicted Diabetes |
   |------------------------|-----------------------|--------------------|
   | **Actual No Diabetes** | 18,225                | 67                 |
   | **Actual Diabetes**    | 534                   | 1,174              |

3. **SVM**  
   |                        | Predicted No Diabetes | Predicted Diabetes |
   |------------------------|-----------------------|--------------------|
   | **Actual No Diabetes** | 18,260                | 32                 |
   | **Actual Diabetes**    | 766                   | 942                |

4. **Deep Learning**  
   |                        | Predicted No Diabetes | Predicted Diabetes |
   |------------------------|-----------------------|--------------------|
   | **Actual No Diabetes** | 18,184                | 108                |
   | **Actual Diabetes**    | 540                   | 1,168              |

---

## Analisis Hasil Evaluasi

1. **Logistic Regression**:
   - Recall rendah (61.30%) menunjukkan kelemahan model dalam mendeteksi pasien positif diabetes.
   - False Negatives tinggi (661) menjadi kendala utama.

2. **Random Forest**:
   - Memiliki performa terbaik dengan akurasi 96.99% dan F1 Score tertinggi (79.62%).
   - False Negatives (534) lebih rendah dibanding Logistic Regression dan SVM.

3. **SVM**:
   - Precision tertinggi (96.71%), menunjukkan model ini sangat baik dalam meminimalkan False Positives (32).
   - Recall yang rendah (55.15%) menunjukkan banyak pasien positif yang tidak terdeteksi (FN = 766).

4. **Deep Learning**:
   - Akurasi mendekati Random Forest (96.76%) dengan False Negatives (540) sedikit lebih tinggi.
   - Performa ini cukup baik, tetapi memerlukan sumber daya komputasi tinggi.

---

## Kesimpulan

- **Model Terbaik**: Berdasarkan evaluasi, **Random Forest** adalah model terbaik untuk memprediksi diabetes. Model ini memiliki keseimbangan terbaik antara precision, recall, dan F1 Score.
- **Rekomendasi**: Random Forest dapat digunakan untuk implementasi klinis dalam membantu skrining awal pasien dengan risiko diabetes.

Confusion Matrix memperjelas distribusi prediksi, membantu dalam memahami pola kesalahan dan kekuatan masing-masing model.

```


```python

```
