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

## Data Understanding

### Deskripsi Dataset
Dataset yang digunakan dalam proyek ini berasal dari [Kaggle - Diabetes Prediction Dataset](https://www.kaggle.com/code/tumpanjawat/diabetes-eda-random-forest-hp?select=diabetes_prediction_dataset.csv). Dataset ini berisi informasi terkait kondisi pasien yang dapat digunakan untuk memprediksi apakah seseorang memiliki diabetes atau tidak. Dataset ini memiliki **100.000 data** dengan **9 kolom**.

### Informasi Dataset
Informasi dataset diberikan menggunakan fungsi `data.info()`. Berikut adalah hasilnya:

| Kolom                  | Tipe Data | Jumlah Data | Deskripsi                                                                 |
|------------------------|-----------|-------------|---------------------------------------------------------------------------|
| gender                | object    | 100,000     | Jenis kelamin pasien.                                                    |
| age                   | float64   | 100,000     | Usia pasien dalam tahun.                                                 |
| hypertension          | int64     | 100,000     | Apakah pasien memiliki hipertensi (0: Tidak, 1: Ya).                     |
| heart_disease         | int64     | 100,000     | Apakah pasien memiliki penyakit jantung (0: Tidak, 1: Ya).               |
| smoking_history       | object    | 100,000     | Riwayat merokok pasien (current, former, never, dll.).                   |
| bmi                   | float64   | 100,000     | Indeks Massa Tubuh pasien.                                               |
| HbA1c_level           | float64   | 100,000     | Tingkat HbA1c (kadar gula darah dalam 3 bulan terakhir).                 |
| blood_glucose_level   | int64     | 100,000     | Tingkat glukosa darah pasien.                                            |
| diabetes              | int64     | 100,000     | Target prediksi (0: Tidak Diabetes, 1: Diabetes).                        |

Dataset **tidak memiliki missing values**, sehingga dapat langsung digunakan untuk proses analisis dan pemodelan.

### Statistik Deskriptif
Berikut adalah statistik deskriptif untuk fitur numerik dalam dataset:

| Fitur                  | Count      | Mean      | Std Dev   | Min       | 25%       | 50%       | 75%       | Max       |
|------------------------|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| age                   | 100,000    | 41.89     | 22.52     | 0.08      | 24.00     | 43.00     | 60.00     | 80.00     |
| hypertension          | 100,000    | 0.07      | 0.26      | 0.00      | 0.00      | 0.00      | 0.00      | 1.00      |
| heart_disease         | 100,000    | 0.04      | 0.19      | 0.00      | 0.00      | 0.00      | 0.00      | 1.00      |
| bmi                   | 100,000    | 27.32     | 6.64      | 10.01     | 23.63     | 27.32     | 29.58     | 95.69     |
| HbA1c_level           | 100,000    | 5.53      | 1.07      | 3.50      | 4.80      | 5.80      | 6.20      | 9.00      |
| blood_glucose_level   | 100,000    | 138.06    | 40.71     | 80.00     | 100.00    | 140.00    | 159.00    | 300.00    |

### Exploratory Data Analysis (EDA)

#### 1. Distribusi Usia
Distribusi usia menunjukkan bahwa dataset mencakup rentang usia yang luas, dari bayi hingga lansia. Sebagian besar data berpusat pada usia 40-60 tahun.

#### 2. Distribusi BMI
Indeks Massa Tubuh (BMI) mayoritas berada dalam rentang normal, tetapi ada nilai ekstrim yang signifikan. 

#### 3. Hubungan Fitur dengan Diabetes
- **Hipertensi dan Diabetes**: Pasien dengan hipertensi lebih cenderung memiliki diabetes dibandingkan yang tidak memiliki hipertensi.
- **HbA1c Level dan Glukosa Darah**: Tingkat HbA1c dan glukosa darah menunjukkan hubungan yang kuat dengan diabetes.
- **Gender dan Riwayat Merokok**: Tidak menunjukkan pengaruh signifikan terhadap prediksi diabetes.

### Kesimpulan Data Understanding
- Dataset ini sudah bersih tanpa missing values.
- Variabel kategori (gender, smoking_history) perlu diubah menjadi numerik melalui proses encoding.
- Fitur numerik seperti **age**, **bmi**, dan **blood_glucose_level** menunjukkan distribusi normal dengan beberapa outlier yang perlu diperhatikan.
- Data siap digunakan untuk tahap preprocessing dan modeling.

## Data Preparation

Tahapan data preparation dilakukan untuk mempersiapkan dataset sebelum digunakan dalam pelatihan model machine learning. Berikut langkah-langkahnya:

### 1. Konversi Variabel Kategori menjadi Variabel Dummy
Dataset memiliki kolom kategori seperti `gender` dan `smoking_history`, yang perlu diubah menjadi format numerik. Proses ini dilakukan menggunakan **one-hot encoding** dengan fungsi `pd.get_dummies()`.

Hasil konversi:
- Kolom `gender` diubah menjadi `gender_Male` dan `gender_Other`.
- Kolom `smoking_history` diubah menjadi beberapa kolom dummy seperti `smoking_history_current`, `smoking_history_ever`, dan lainnya.

Proses ini memastikan bahwa model machine learning dapat memproses semua data dalam format numerik.

---

### 2. Pemisahan Fitur dan Target
Dataset dipisahkan menjadi dua bagian utama:
- **Fitur (X)**: Semua kolom kecuali target (`diabetes`).
- **Target (y)**: Kolom `diabetes` sebagai label target untuk klasifikasi.

Hasil pemisahan memastikan bahwa input data (X) dan output target (y) dapat digunakan secara terpisah dalam proses pelatihan model.

---

### 3. Normalisasi Data
Normalisasi data dilakukan menggunakan **Min-Max Scaler** untuk memastikan semua fitur numerik berada dalam skala yang sama (0 hingga 1). Hal ini penting untuk menghindari bias dalam model yang disebabkan oleh perbedaan skala antar fitur.

Contoh hasil normalisasi:
- Fitur seperti `age`, `bmi`, dan `blood_glucose_level` diubah ke rentang nilai 0 hingga 1.

---

### 4. Pembagian Data Latih dan Data Uji
Dataset dibagi menjadi dua bagian:
- **Data latih (80%)**: Digunakan untuk melatih model.
- **Data uji (20%)**: Digunakan untuk mengevaluasi performa model.

Proses pembagian dilakukan dengan memastikan distribusi target tetap seimbang. Distribusi target pada data latih dan uji sebagai berikut:
- **Data latih**: 91.5% Non-Diabetes, 8.5% Diabetes.
- **Data uji**: 91.4% Non-Diabetes, 8.6% Diabetes.

Hasil pembagian data ini memastikan bahwa model dapat dilatih dan diuji secara adil dengan distribusi data yang representatif.

---

### Kesimpulan Data Preparation
Setelah melalui proses ini, dataset siap digunakan untuk proses modeling:
1. Variabel kategori telah dikonversi menjadi variabel dummy.
2. Semua fitur numerik telah dinormalisasi ke rentang nilai 0 hingga 1.
3. Dataset telah dibagi menjadi data latih dan uji dengan distribusi target yang seimbang.

Dataset yang telah dipersiapkan ini memastikan kualitas data yang optimal untuk pelatihan dan evaluasi model machine learning.

## Modeling

Pada tahap ini, dilakukan pemodelan data menggunakan berbagai algoritma machine learning untuk memprediksi diabetes. Model yang digunakan meliputi Logistic Regression, Random Forest, Support Vector Machine (SVM), dan Deep Learning. Setiap model dievaluasi berdasarkan metrik akurasi, precision, recall, dan F1-score.

---

### 1. Logistic Regression
Logistic Regression adalah model yang sederhana namun efektif untuk masalah klasifikasi biner. Model ini menggunakan fungsi logistik untuk memprediksi probabilitas sebuah data termasuk dalam kelas tertentu.

- **Parameter**: 
  - `random_state=42` untuk memastikan hasil yang konsisten.
- **Hasil Evaluasi**:
  - **Akurasi**: 0.96
  - **Precision**: 0.87
  - **Recall**: 0.61
  - **F1 Score**: 0.72

**Kelebihan**:
- Cepat dalam pelatihan dan prediksi.
- Mudah diinterpretasikan.

**Kekurangan**:
- Tidak bekerja optimal jika data memiliki pola yang kompleks atau non-linear.

#### **Cara Kerja**:
1. Model mempelajari hubungan antara fitur input dan target selama pelatihan.
2. Probabilitas diprediksi untuk setiap data.
3. Jika probabilitas lebih tinggi dari ambang batas (default 0.5), data diklasifikasikan sebagai positif (diabetes). Jika tidak, diklasifikasikan sebagai negatif (tidak diabetes).
---

### 2. Random Forest
Random Forest adalah model ensemble yang menggabungkan banyak pohon keputusan untuk menghasilkan prediksi yang lebih akurat dan stabil.

- **Parameter**:
  - `random_state=42` untuk konsistensi hasil.
- **Hasil Evaluasi**:
  - **Akurasi**: 0.97
  - **Precision**: 0.94
  - **Recall**: 0.69
  - **F1 Score**: 0.80

**Kelebihan**:
- Menangani fitur yang tidak seimbang dengan baik.
- Resisten terhadap overfitting dibandingkan model pohon tunggal.

**Kekurangan**:
- Membutuhkan lebih banyak sumber daya komputasi.

#### **Cara Kerja**:
1. Dataset dibagi menjadi beberapa subset secara acak.
2. Setiap subset digunakan untuk melatih pohon keputusan independen.
3. Prediksi akhir ditentukan melalui voting mayoritas dari semua pohon.
---

### 3. Support Vector Machine (SVM)
SVM adalah algoritma yang mencoba memisahkan data dengan hyperplane optimal di ruang fitur.

- **Parameter**:
  - `random_state=42` untuk konsistensi.
- **Hasil Evaluasi**:
  - **Akurasi**: 0.96
  - **Precision**: 0.97
  - **Recall**: 0.55
  - **F1 Score**: 0.70

**Kelebihan**:
- Baik untuk data yang berskala kecil dengan jumlah fitur yang banyak.
- Memiliki margin yang jelas antara kelas.

**Kekurangan**:
- Tidak efisien pada dataset besar.
- Memerlukan tuning parameter yang kompleks.

#### **Cara Kerja**:
1. SVM mencari hyperplane yang memisahkan dua kelas dengan jarak terbesar (margin) dari data terdekat.
2. Jika data tidak dapat dipisahkan secara langsung, SVM menggunakan kernel untuk mengubah data ke dimensi yang lebih tinggi, di mana pemisahan dapat dilakukan.
---

### 4. Deep Learning
Model deep learning menggunakan **Neural Network** dengan arsitektur sederhana yang terdiri dari beberapa lapisan.

- **Arsitektur**:
  - Input layer: Menggunakan jumlah fitur pada data.
  - Hidden layers: Dua hidden layer dengan 64 dan 32 neuron.
  - Output layer: Satu neuron dengan fungsi aktivasi sigmoid.
- **Hasil Evaluasi**:
  - **Akurasi**: 0.97
  - **Precision**: 0.93
  - **Recall**: 0.68
  - **F1 Score**: 0.79

**Kelebihan**:
- Mampu menangkap pola data yang kompleks.
- Skalabilitas untuk dataset besar.

**Kekurangan**:
- Membutuhkan waktu pelatihan yang lebih lama.
- Interpretasi model lebih sulit dibanding model lainnya.

#### **Cara Kerja**:
1. Data masuk melalui lapisan input dan diproses melalui beberapa lapisan tersembunyi.
2. Setiap lapisan menerapkan fungsi aktivasi untuk menangkap pola non-linear dalam data.
3. Lapisan output memberikan prediksi probabilitas.
4. Model dilatih melalui proses iteratif untuk meminimalkan kesalahan prediksi.

---

### 5. Perbandingan Hasil Model
Hasil evaluasi semua model dirangkum dalam tabel berikut:

| Model               | Akurasi | Precision | Recall | F1 Score |
|---------------------|---------|-----------|--------|----------|
| Logistic Regression | 0.95920 | 0.871048  | 0.612998 | 0.719588 |
| Random Forest       | 0.96995 | 0.946011  | 0.687354 | 0.796202 |
| SVM                 | 0.96010 | 0.967146  | 0.551522 | 0.702461 |
| Deep Learning       | 0.96960 | 0.935252  | 0.685012 | 0.790808 |

---

### 6. Model Terbaik
Dari hasil evaluasi, model **Random Forest** memiliki performa terbaik berdasarkan keseimbangan antara akurasi, precision, recall, dan F1-score. Model ini dipilih sebagai model akhir karena performanya yang unggul dan kemampuannya menangani dataset yang kompleks.

## Evaluation

Pada tahap ini, dilakukan evaluasi terhadap performa model berdasarkan berbagai metrik, yaitu **Akurasi**, **Precision**, **Recall**, dan **F1 Score**. Selain itu, analisis confusion matrix digunakan untuk memahami prediksi yang benar dan salah dari model.

---

### Metrik Evaluasi

1. **Akurasi**: Mengukur proporsi prediksi yang benar dari total prediksi.
2. **Precision**: Mengukur ketepatan prediksi kelas positif dari semua prediksi positif.
3. **Recall**: Mengukur sensitivitas model dalam mendeteksi kelas positif.
4. **F1 Score**: Rata-rata harmonis antara precision dan recall.

---

### Hasil Evaluasi

Berikut adalah hasil evaluasi dari keempat model yang digunakan:

| Model               | Akurasi | Precision | Recall | F1 Score |
|---------------------|---------|-----------|--------|----------|
| Logistic Regression | 0.95920 | 0.871048  | 0.612998 | 0.719588 |
| Random Forest       | 0.96995 | 0.946011  | 0.687354 | 0.796202 |
| SVM                 | 0.96010 | 0.967146  | 0.551522 | 0.702461 |
| Deep Learning       | 0.96960 | 0.935252  | 0.685012 | 0.790808 |

Dari tabel di atas, model Random Forest memiliki performa terbaik secara keseluruhan dengan nilai **F1 Score** tertinggi.

---

### Analisis Confusion Matrix

Confusion matrix memberikan informasi tentang prediksi benar (True Positives dan True Negatives) serta prediksi salah (False Positives dan False Negatives). Berikut adalah confusion matrix dari masing-masing model:

![confusion matrik](https://github.com/user-attachments/assets/acea79b3-6a4d-40ef-a5e0-7cd3af38e449)
Hasil Confusion Matrix

### 1. Logistic Regression
Confusion Matrix untuk model Logistic Regression menunjukkan:
- **True Positive (TP)**: 1047, prediksi diabetes benar.
- **True Negative (TN)**: 18137, prediksi tidak diabetes benar.
- **False Positive (FP)**: 155, prediksi diabetes salah.
- **False Negative (FN)**: 661, prediksi tidak diabetes salah.

Analisis:
- Logistic Regression memiliki akurasi sebesar **95.92%**. Namun, recall cukup rendah (**61.30%**) yang menunjukkan bahwa model ini tidak sepenuhnya efektif dalam mendeteksi kasus diabetes.
- 
---
### 2. Random Forest
Confusion Matrix untuk model Random Forest menunjukkan:
- **True Positive (TP)**: 1174, prediksi diabetes benar.
- **True Negative (TN)**: 18225, prediksi tidak diabetes benar.
- **False Positive (FP)**: 67, prediksi diabetes salah.
- **False Negative (FN)**: 534, prediksi tidak diabetes salah.

Analisis:
- Random Forest menunjukkan kinerja yang lebih baik dibandingkan Logistic Regression dengan akurasi **96.99%** dan F1 Score **79.62%**. Recall meningkat menjadi **68.73%**, yang menunjukkan peningkatan dalam mendeteksi kasus diabetes.
- 
---
### 3. Support Vector Machine (SVM)
Confusion Matrix untuk model SVM menunjukkan:
- **True Positive (TP)**: 942, prediksi diabetes benar.
- **True Negative (TN)**: 18260, prediksi tidak diabetes benar.
- **False Positive (FP)**: 32, prediksi diabetes salah.
- **False Negative (FN)**: 766, prediksi tidak diabetes salah.

Analisis:
- SVM memiliki akurasi **96.01%**, tetapi recall **55.15%** yang cukup rendah dibandingkan Random Forest. Ini menunjukkan bahwa SVM lebih baik dalam menghindari false positives daripada mendeteksi kasus diabetes.

---
### 4. Deep Learning
Confusion Matrix untuk model Deep Learning menunjukkan:
- **True Positive (TP)**: 1170, prediksi diabetes benar.
- **True Negative (TN)**: 18211, prediksi tidak diabetes benar.
- **False Positive (FP)**: 81, prediksi diabetes salah.
- **False Negative (FN)**: 538, prediksi tidak diabetes salah.

Analisis:
- Deep Learning menunjukkan kinerja akurasi **96.89%** dan F1 Score **79.08%**, mendekati Random Forest. Namun, model ini memiliki performa yang lebih stabil berdasarkan metrik evaluasi.

---
### Kesimpulan
1. **Model terbaik**: Random Forest.
   - Alasan: Random Forest memiliki keseimbangan terbaik antara Akurasi (0.97), Precision (0.94), Recall (0.69), dan F1 Score (0.80).
2. Logistic Regression dan Deep Learning juga memiliki performa yang baik, namun sedikit kalah dibandingkan Random Forest.
3. SVM memiliki precision tinggi (0.97) namun recall rendah (0.55), sehingga kurang baik dalam mendeteksi kasus positif diabetes.

Model Random Forest dipilih untuk digunakan pada tahap implementasi karena performa terbaiknya dalam mendeteksi diabetes berdasarkan dataset yang digunakan.

