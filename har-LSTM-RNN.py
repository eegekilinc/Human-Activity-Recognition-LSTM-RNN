import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf  # BURASI DÜZELTİLDİ (eskiden 'as pd' idi)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Dropout
from tensorflow.keras.utils import to_categorical

# 1. VERİ YÜKLEME
# ---------------------------------------------------------
print("Veriler yükleniyor...")
# train.csv dosyasının tam adını buraya yazdığınızdan emin olun
try:
    train_df = pd.read_csv('train.csv')
    print("Train verisi başarıyla yüklendi.")
except FileNotFoundError:
    print("HATA: train.csv dosyası bulunamadı! Lütfen dosyanın kodla aynı klasörde olduğundan emin olun.")
    exit()

# Test.csv kontrolü
try:
    test_df = pd.read_csv('test.csv')
    print("Test verisi başarıyla yüklendi.")
except FileNotFoundError:
    print("Test dosyası bulunamadı, Train verisi bölünüyor...")
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(train_df, test_size=0.3, random_state=42)

# 2. VERİ ÖN İŞLEME
# ---------------------------------------------------------
print("Veri ön işleme yapılıyor...")
# Girdi (X) ve Hedef (y) ayrımı
X_train = train_df.drop(['subject', 'Activity'], axis=1).values
y_train = train_df['Activity']

X_test = test_df.drop(['subject', 'Activity'], axis=1).values
y_test = test_df['Activity']

# Etiketleri Sayısallaştırma (String -> Integer -> One-Hot)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

y_train_final = to_categorical(y_train_enc)
y_test_final = to_categorical(y_test_enc)

# Veriyi Ölçeklendirme (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RNN ve LSTM için Boyut Değiştirme (Reshape)
# Keras RNN/LSTM katmanları [Samples, Time Steps, Features] formatı ister.
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

print(f"Eğitim Verisi Boyutu: {X_train_reshaped.shape}")
print(f"Sınıf Sayısı: {len(le.classes_)}")

# 3. MODELLERİN KURULUMU
# ---------------------------------------------------------

# --- Model 1: Simple RNN ---
print("\nRNN Modeli Eğitiliyor (Bu işlem biraz sürebilir)...")
model_rnn = Sequential()
model_rnn.add(SimpleRNN(64, input_shape=(1, X_train_reshaped.shape[2]), activation='relu'))
model_rnn.add(Dropout(0.2))
model_rnn.add(Dense(32, activation='relu'))
model_rnn.add(Dense(len(le.classes_), activation='softmax'))

model_rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_rnn = model_rnn.fit(X_train_reshaped, y_train_final, epochs=15, batch_size=64, validation_data=(X_test_reshaped, y_test_final), verbose=1)

# --- Model 2: LSTM ---
print("\nLSTM Modeli Eğitiliyor (Bu işlem biraz sürebilir)...")
model_lstm = Sequential()
model_lstm.add(LSTM(64, input_shape=(1, X_train_reshaped.shape[2]), activation='relu'))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(32, activation='relu'))
model_lstm.add(Dense(len(le.classes_), activation='softmax'))

model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_lstm = model_lstm.fit(X_train_reshaped, y_train_final, epochs=15, batch_size=64, validation_data=(X_test_reshaped, y_test_final), verbose=1)

# 4. GÖRSELLEŞTİRME VE RAPORLAMA
# ---------------------------------------------------------
print("\nGrafikler oluşturuluyor...")

# Grafik 1: Accuracy ve Loss Karşılaştırması
plt.figure(figsize=(14, 6))

# Accuracy Subplot
plt.subplot(1, 2, 1)
plt.plot(history_rnn.history['accuracy'], label='RNN Train', linestyle='--')
plt.plot(history_rnn.history['val_accuracy'], label='RNN Test')
plt.plot(history_lstm.history['accuracy'], label='LSTM Train', linestyle='--')
plt.plot(history_lstm.history['val_accuracy'], label='LSTM Test')
plt.title('Model Doğruluk (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Subplot
plt.subplot(1, 2, 2)
plt.plot(history_rnn.history['loss'], label='RNN Train', linestyle='--')
plt.plot(history_rnn.history['val_loss'], label='RNN Test')
plt.plot(history_lstm.history['loss'], label='LSTM Train', linestyle='--')
plt.plot(history_lstm.history['val_loss'], label='LSTM Test')
plt.title('Model Kayıp (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('accuracy_loss_grafigi.png') 
print("- accuracy_loss_grafigi.png kaydedildi.")

# Grafik 2: Confusion Matrix (LSTM Modeli İçin)
y_pred_probs = model_lstm.predict(X_test_reshaped)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_final, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix (LSTM Modeli)')
plt.ylabel('Gerçek Sınıf')
plt.xlabel('Tahmin Edilen Sınıf')
plt.savefig('confusion_matrix.png')
print("- confusion_matrix.png kaydedildi.")

# Sonuç Raporu Metni
print("\n" + "="*40)
print("--- PROJE SONUÇLARI ---")
print(f"RNN Test Accuracy: {np.max(history_rnn.history['val_accuracy']):.4f}")
print(f"LSTM Test Accuracy: {np.max(history_lstm.history['val_accuracy']):.4f}")
print("\nSınıflandırma Raporu (LSTM):")
print(classification_report(y_true, y_pred, target_names=le.classes_))
print("="*40)
print("İşlem tamamlandı. Klasörünüzdeki PNG dosyalarını kontrol ediniz.")