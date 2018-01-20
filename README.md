# Face Recognition Sikemas
Ini adalah modul untuk face recognition Sikemas IF (Sistem Kehadiran Mahasiswa Informatika ITS) menggunakan [OpenFace](https://cmusatyalab.github.io/openface/setup/), [Torch](http://torch.ch/docs/getting-started.html#_) dan [Scikit-Learn](http://scikit-learn.org/stable/install.html). Pastikan ketiga library itu sudah terinstall pada sistem.

## Langkah-Langkah Instalasi

 1. Clone repository ini
 2. Download PreTrained [shape_predictor_68_face_landmarks.dat](https://drive.google.com/open?id=1su4VCnrAAlttST_qlwd2vM6VyXQ-ipos) dan letakkan pada direktori yang sama dengan repository ini.
 3. Download PreTrained Convolutional Neural Networks [nn4.small2.v1.t7](https://drive.google.com/open?id=13B_M3gS7CzCF6Zgi1UA8iXdBGEW9uKzC) dan letakkan pada direktori yang sama dengan repository ini.
 4. Letakkan direktori master OpenFace pada direktori yang sama juga.
 
 ## Langkah-Langkah Training
 **Penting:** Setiap kelas akan memiliki model klasifikasinya sendiri sehingga proses training akan sebanyak jumlah kelas yang menggunakan Sikemas. Ikuti langkah-langkahnya sebagai berikut:
 
 1. Pada direktori yang sama dengan instalasi modul ini, buat direktori dengan nama **kelas** dengan menjalankan `sudo mkdir kelas` pada terminal.
 2. Jalankan `sudo ./create.sh PBKK-A` untuk proses training pada kelas **PBKK-A**.
 3. Masuk ke direktori PBKK-A dengan `cd kelas/PBKK-A`. Pada direktori ini, harusnya tedapat direktori `PBKK-A/training-images/` hasil dari script bash pada langkah 2.
 4. Copy semua data gambar mahasiswa pada direktori `PBKK-A/training-images/`. Tentunya berisi data mahasiswa yang mengikuti kelas PBKK-A. Setiap data gambar mahasiswa dijadikan satu direktori dan direktorinya dinamai dengan **NRP mahasiswa** tersebut. Sebagai contoh:
	 - `PBKK-A/training-images/5114100177`
	 - `PBKK-A/training-images/5114100024`
	 - `PBKK-A/training-images/5114100001` 
 5. Pada direktori `PBKK-A` lakukan training wajah dengan menjalankan `./train.sh` tanpa menggunakan `sudo`.
 6. Hasilnya adalah file `model.pkl`yang merupakan model hasil proses training.

## Algoritma Training
Algoritma training yang digunakan adalah **Random Forest** dengan `n_estimators=1000`. Model ini mencapai akurasi 100% pada data training yang diujicobakan menggunakan modul ini.

## Source
https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
