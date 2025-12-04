# Human-Activity-Recognition-LSTM-RNN

## Proje Açıklaması
Bu depo, akıllı telefonlardan toplanan ivmeölçer ve jiroskop sensör verilerini kullanarak altı farklı insan aktivitesini (Yürüme, Oturma, Ayakta Durma vb.) yüksek doğrulukla sınıflandırmayı amaçlayan bir Derin Öğrenme projesini içermektedir.
Projede **Tekrarlayan Sinir Ağları (RNN)** ve **Uzun Kısa Süreli Bellek (LSTM)** mimarileri karşılaştırmalı olarak uygulanmıştır.

## Elde Edilen Başarı
* **Veri Seti:** Human Activity Recognition (HAR)
* **En Yüksek Doğruluk (LSTM):** %95.01

---

## 🛠️ Kullanılan Teknolojiler
* Python
* TensorFlow / Keras (Derin Öğrenme Modeli)
* Pandas & NumPy
* Scikit-learn (Veri Ön İşleme)

## 📂 Dosya Yapısı
* `har-LSTM-RNN.py`: Projeyi çalıştıran ve modelleri eğiten ana Python kodu.
* `train.csv / test.csv`: Modelin eğitildiği ve test edildiği veri setleri.
* `accuracy_loss_grafigi.png`: Eğitim ve doğrulama süreçlerindeki performans eğrileri.
* `confusion_matrix.png`: LSTM modelinin sınıf bazlı doğruluk performansını gösteren görsel.

## ▶️ Nasıl Çalıştırılır?
Projenin yerel makinenizde çalıştırılması için aşağıdaki adımları izleyin:

1.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
    ```
2.  `train.csv` ve `test.csv` dosyalarını kodla aynı klasöre yerleştirin.
3.  Terminalde kodu çalıştırın:
    ```bash
    python har-LSTM-RNN.py 
    ```

## 📈 Sonuç Görselleri
Proje dosyası içerisindeki görseller, modelin test verisi üzerindeki başarımını göstermektedir:
