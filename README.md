# skripsi
Skripsi Klasifikasi Tipe Vocal Suara dengan Menggunakan STFT
Pada repisotory ini berisikan kode - kode yang digunakan dalam pembuatan model dan ekstraksi fitur untuk projek skripsi klasifikasi tipe suara vokal. Di dalam repo ini terdapat notebook ipynb untuk ekstraksi fitur dan pelatihan model CNN, Source Code GUI, File JSON dari dataset yang sudah diekstrak, dan model h5 untuk klasifikasi.

Ada dua model pada repo ini, keduanya memiliki akurasi yang sama namun memilkiki kemampuan untuk memprediksi kelas yang berbeda
- Model VocalClassifier10.h5 dapat memprediksi kelas sopran, bass, dan tenor dengan baik namun alto masih kurang
- Model VocalClassifierAugmented4.h5 dapat memprediksi kelas sopran, bass, dan alto dengan baik namun tenor terkadang masih sulit untuk didapatkan
