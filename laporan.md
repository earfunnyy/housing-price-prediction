# Laporan Proyek Machine Learning - Muhammad Daffa Irfani
## Domain Proyek
Proyek ini akan membuat sebuah model untuk prediksi harga rumah menggunakan dataset berisikan fitur terkait karakteristik rumah. Masalah ini sangat penting karena penetapan harga properti seperti rumah memiliki dampak besar bagi stabilitas pasar dan pertumbuhan ekonomi di sebuah wilayah. Harga rumah dapat sangat bervariasi secara signifikan karena banyak faktor seperti lokasi, ukuran, dan fasilitas. Dengan membuat sebuah model, kita dapat membantu membuat keputusan dalam menetapkan harga properti ini.
Model yang dibangun akan membantu mempercepat transaksi properti dan menciptakan pemetaan harga yang lebih adil di pasar properti rumah. Masalah ini sangat penting karena dapat meningkatkan transparansi dan efisiensi pasar. Prediksi harga rumah menggunakan model machine learning memungkinkan kita untuk menggabungkan banyak variabel yang sulit digunakan dalam penilaian tradisional.
## Business Understanding
### Pernyataan Masalah
1. Bagaimana model memprediksi harga rumah dengan akurat berdasarkan fitur-fitur yang ada?
2. Apa saja faktor yang paling berpengaruh terhadap harga rumah?
3. Bagaimana meningkatkan akurasi model dalam memprediksi harga rumah?

### Goals
1. Membangun model yang dapat memprediksi harga rumah berdasarkan fitur-fitur yang ada.
2. Mengidentifikasi fitur kunci yang memengaruhi harga rumah.
3. Mengoptimalkan model untuk meningkatkan akurasi prediksi.

### Pernyataan Solusi
1. Model akan dibuat dengan menggunakan Random Forest Regressor untuk memprediksi harga rumah, karena model ini dapat menangani hubungan non-linear dan bekerja dengan baik degan data tabular.
2. Model yang dibuat akan dibandingkan kinerjanya dengan model lain seperti Regresi Linear.
3. Kami akan melakukan hyperparameter tuning untuk mengoptimalkan kinerja model, dengan menggunakan metrik seperti Mean Squared error (MSE) dan R-squared.


## Data Understanding
Dataset yang digunakan dalam proyek ini terdiri dari 545 data dan 13 kolom. Dataset ini berasal dari [Kaggle](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction/code) . Fitur-fitur kunci dalam dataset ini adalah:
- Price: The price of the house.
- Area: The total area of the house in square feet.
- Bedrooms: The number of bedrooms in the house.
- Bathrooms: The number of bathrooms in the house.
- Stories: The number of stories in the house.
- Mainroad: Whether the house is connected to the main road (Yes/No).
- Guestroom: Whether the house has a guest room (Yes/No).
- Basement: Whether the house has a basement (Yes/No).
- Hot water heating: Whether the house has a hot water heating system (Yes/No).
- Airconditioning: Whether the house has an air conditioning system (Yes/No).
- Parking: The number of parking spaces available within the house.
- Prefarea: Whether the house is located in a preferred area (Yes/No).
- Furnishing status: The furnishing status of the house (Fully Furnished, Semi-Furnished, Unfurnished).

### Insight Data
Setelah melakukan Exploratory Data Analysis (EDA), data yang dimiliki sudah sangat bersih seperti tidak adanya nilai 0 dan juga outlier sehingga tidak diperlukan operasi Data Cleaning.

## Data Preparation
Langkah-langkah pada tahp ini meliputi:
1. One-Hot Encoding: Variabel kategori "Furnishingstatus" akan di-encode menggunakan teknik one-hot encoding.
2. Konversi fitur biner: Variabel kategori biner (misalnya, Mainroad) akan dikonversi menjadi 1/0.
3. Feature Scaling: Variabel yang diperlukan akan dinormalisasi menggunakan StandardScaler.
4. Train-test split: Dataset dipisah menjadi dataset latih dan juga tes sebelum dimasukkan ke dalam model dengan proporsi 80:20.

## Modeling
Kami akan menggunakan beberapa model machine learning untuk menyelesaikan masalah ini, termasuk:
1. Random Forest Regressor : Model ini kuat dalam menangani data kategori dan kontinu yang dimiliki oleh dataset. Model ini juga menjadi model utama.
2. Regresi Linear: Model ini merupakan model sederhana yang mudah diinterpretasikan sebagai alat pembanding dengan model utama.

Random Forest Regressor adalah algoritma pembelajaran mesin berbasis ensemble yang digunakan untuk tugas regresi seperti pada prediksi harga rumah. Algoritma ini bekerja dengan membuat beberapa decision trees selama pelatihan dan menggabungkan hasil prediksi dari setiap pohon untuk menghasilkan prediksi akhir yang lebih akurat.

### Cara Kerja Random Forest
#### 1. Membangun pohon keputusan
Algoritma membuat beberapa pohon keputusan dari subset data latih. Setiap pohon dibuat menggunakan subset data yang berbeda dan fitur yang dipilih secara acak, sehingga setiap pohon memiliki sedikit perbedaan.

#### 2. Prediksi
Pada kasus ini yaitu regresi, Random Forest menggabungkan hasil prediksi dari semua pohon dengan menghitung rata-rata dari semua output pohon.

#### 3. Mekanisme Voting dan Averaging
Dalam regresi, hasil akhir atau keputusan yang dibuat adalah rata-rata dari prediksi yang dihasilkan oleh semua pohon.


### Parameter Utama dalam Random Forest Regressor
**n_estimators**: Jumlah pohon keputusan dalam hutan. Semakin banyak pohon, semakin stabil model. Contoh: n_estimators=100.
**max_depth**: Kedalaman maksimum setiap pohon. Semakin dalam pohon, semakin banyak detail yang ditangkap, tetapi dapat menyebabkan overfitting. Contoh: max_depth=None (artinya pohon akan tumbuh hingga sempurna).
**min_samples_split**: Jumlah minimum sampel yang diperlukan untuk memecah node internal. Contoh: min_samples_split=2.
**min_samples_leaf**: Jumlah minimum sampel yang diperlukan pada node daun. Contoh: min_samples_leaf=1.
**max_features**: Jumlah maksimum fitur yang dipertimbangkan saat mencari split terbaik. Contoh: max_features='auto' atau sqrt.
**random_state**: Seed untuk menghasilkan hasil yang dapat direproduksi. Contoh: random_state=42.


### Keunggulan Random Forest
- Tahan Terhadap Overfitting: Dengan menggabungkan hasil dari banyak pohon, Random Forest cenderung tidak overfit pada data latih.
- Robust terhadap Outlier: Random Forest tidak terlalu dipengaruhi oleh outlier karena hasilnya merupakan rata-rata dari banyak pohon.


### Pelatihan model
Kami akan mulai dengan melatih dan mengevaluasi model-model ini menggunakan hyperparameter default. Kinerja akan diukur dengan metrik Mean Squared Error (MSE) dan R-squared.

### Tuning model
Setelah membuat baseline model, kami melakukan hyperparameter tuning menggunakan RandomizedSearchCV untuk meningkatkan kinerja model.

### Penjelasan Proses Tuning Random Forest Regressor dengan RandomizedSearchCV
Pada kode yang kami buat, proses tuning dilakukan dengan menggunakan `RandomizedSearchCV`, yang mencoba berbagai kombinasi hyperparameter secara acak dari ruang pencarian yang telah ditentukan sebelumnya. Tujuan utama tuning ini adalah untuk menemukan kombinasi hyperparameter yang menghasilkan model dengan kinerja terbaik pada data latih.

#### Proses Tuning dan Parameter yang Dicoba

1. **Definisi Parameter untuk Tuning:**
   Kamu telah mendefinisikan beberapa parameter utama yang memengaruhi kinerja model `RandomForestRegressor`. Parameter ini termasuk:
   - **`n_estimators`**: Jumlah pohon dalam hutan. kami mengujikan nilai 100, 200, dan 500.
   - **`max_depth`**: Kedalaman maksimum pohon. Nilai yang diuji meliputi `None`, 10, 20, dan 30.
   - **`min_samples_split`**: Jumlah minimum sampel yang diperlukan untuk memecah node. Nilai yang diuji adalah 2, 5, dan 10.
   - **`min_samples_leaf`**: Jumlah minimum sampel pada daun node. Nilai yang diuji adalah 1, 2, dan 4.
   - **`max_features`**: Jumlah maksimum fitur yang dipertimbangkan untuk split terbaik. Nilai yang diuji meliputi `'auto'`, `'sqrt'`, dan `'log2'`.

2. **Penggunaan `RandomizedSearchCV`:**
   - **`n_iter=50`**: `RandomizedSearchCV` akan mencoba 50 kombinasi parameter secara acak dari ruang parameter yang telah ditentukan. Ini lebih efisien dibandingkan `GridSearchCV`, yang menguji semua kombinasi.
   - **`cv=3`**: Kode ini menggunakan 3-fold cross-validation untuk mengevaluasi kinerja setiap kombinasi parameter. Ini berarti data dibagi menjadi 3 bagian, dan model dilatih tiga kali, masing-masing dengan bagian yang berbeda sebagai data validasi.
   - **`random_state=42`**: Seed untuk memastikan hasil yang dapat direproduksi.
   - **`n_jobs=-1`**: Semua core CPU yang tersedia akan digunakan untuk mempercepat proses.

3. **Pemilihan Model Terbaik:**
   Setelah proses pencarian selesai, `RandomizedSearchCV` akan menampilkan kombinasi hyperparameter terbaik yang ditemukan selama pencarian. Kami dapat mengakses model terbaik melalui `random_search.best_estimator_` dan mendapatkan model dengan parameter sebagai berikut.
   - n_estimators: 100
    - min_samples_split: 5
    - min_samples_leaf: 1
    - max_features: log2
    - max_depth = 20  

#### Hasil dari Proses Tuning

- **Efisiensi Waktu:** `RandomizedSearchCV` lebih cepat daripada `GridSearchCV` karena hanya mencoba sejumlah kombinasi parameter secara acak, bukan semua kemungkinan kombinasi.
- **Penghindaran Overfitting:** Dengan cross-validation, model diuji pada beberapa subset data, yang membantu dalam mencegah overfitting.

Dengan proses tuning ini, kamu dapat meningkatkan kinerja model `RandomForestRegressor` secara signifikan dengan menemukan pengaturan parameter yang optimal berdasarkan data yang ada.


### Evaluasi Hasil Model

Pada proyek ini, tujuan utama adalah memprediksi harga rumah berdasarkan fitur-fitur yang ada, serta mengidentifikasi fitur-fitur kunci yang berpengaruh terhadap harga tersebut. Model yang dibangun menggunakan Random Forest Regressor telah melalui beberapa tahapan, termasuk baseline model dan hyperparameter tuning. Evaluasi model dilakukan dengan menggunakan metrik Mean Squared Error (MSE) dan R-squared (R²).

**Hasil Akhir:**
- MSE: 0.5396033339784444
- R²: 0.6271963380966454

dengan paramater sebagai berikut:
- n_estimators: 500
- min_samples_split: 2
- min_samples_leaf: 1
- max_features: log2
- max_depth = 10

**Analisis:**
- **Apakah proyek ini berhasil?**  
  Berdasarkan nilai R-squared sebesar 0.6272, model telah mampu menjelaskan sekitar 62.72% dari variabilitas dalam data harga rumah berdasarkan fitur-fitur yang ada. Ini menunjukkan bahwa model cukup baik dalam memprediksi harga rumah, tetapi masih ada ruang untuk peningkatan. Dengan kata lain, model ini berhasil dalam konteks penelitian yang bertujuan untuk menghasilkan prediksi harga rumah, meskipun belum optimal.

- **Apakah hasil evaluasi mencapai tujuan yang diinginkan?**  
  Proyek ini memiliki tujuan untuk membangun model prediktif yang akurat serta mengidentifikasi fitur kunci yang mempengaruhi harga rumah. Model ini telah mencapai tujuan tersebut dengan menghasilkan prediksi yang layak, serta mengidentifikasi fitur-fitur penting seperti jumlah kamar tidur, kamar mandi, dan ketersediaan air panas. Namun, hasil evaluasi menunjukkan bahwa model belum mencapai akurasi yang sempurna, dan peningkatan lebih lanjut bisa dicapai dengan metode yang lebih canggih.

- **Apakah sudah mampu menyelesaikan masalah yang diangkat?**  
  Proyek ini mampu menyelesaikan masalah yang diangkat, yaitu memprediksi harga rumah berdasarkan fitur-fitur yang ada. Dengan model yang dibangun, kita dapat memperkirakan harga rumah dengan tingkat akurasi yang memadai. Meski begitu, ada ruang untuk eksplorasi lebih lanjut seperti penggunaan model-model lain atau tambahan data untuk meningkatkan akurasi prediksi.


### Kesimpulan

Model ini sudah cukup baik dalam melakukan prediksi harga rumah dengan fitur yang dimiliki dengan 5 fitur yang memiliki pengaruh paling tinggi adalah jumlah kamar, ketersediaan air panas, jumlah kamar mandi, airconditioning, dan juga luas area. Ini dibuktikan dengan melatih model terbaik dengan hanya menggunakan 5 fitur saja dan mendapatkan hasil yang mendekati hasil terbaik yaitu R²: 0.5459172970607593 dan MSE: 0.6572482125228639. Hal ini juga menunjukkan bahwa fitur yang dilatih memiliki kesamaan dengan kebutuhan manusia. 

Untuk meningkatkan baseline model yang dibuat, kami melakukan hyperparameter tuning dengan mencari pada 'n estimator', 'min samples split', 'min samples leaf', 'max features', dan juga 'max depth'. model yang dihasilkan telah melalui tahap optimasi menggunakan teknik RandomizedSearchCV untuk mencari hyperparameter terbaik bagi Random Forest Regressor. Setelah dilakukan tuning, model ini menunjukkan peningkatan performa dibandingkan dengan baseline model.

Pada baseline model, R² score adalah sekitar 0.608958183151397 dan MSE adalah 0.5660015972461019. kemudian, setelah dilakukan optimasi hyperparameter, nilai R² meningkat menjadi 0.6271963380966454 dan MSE berkurang menjadi 0.5396033339784444, menunjukkan bahwa model telah menjadi lebih optimal dalam memprediksi harga rumah. Peningkatan ini mengindikasikan bahwa metode optimasi memiliki pengaruh yang baik dalam meningkatkan akurasi prediksi model, sesuai dengan tujuan penelitian.

Dengan demikian, dapat disimpulkan bahwa proyek ini berhasil mengoptimalkan model untuk meningkatkan akurasi prediksi harga rumah secara signifikan dibandingkan dengan baseline model, menjawab problem statement dan goals yang ditetapkan. Dari proses ini, dapat meningkatkan kinerja model dari nilai R-squared 0.608958183151397 menjadi 0.6271963380966454. Dengan proyek ini, kami berhasil membuat sebuah model yang dapat melakukan prediksi harga rumah. Future work juga dapat dilakukan dengan mencoba berbagai model machine learning klasik lainnya maupun deep learning untuk meningkatkan kinerja model pada kasus ini.

