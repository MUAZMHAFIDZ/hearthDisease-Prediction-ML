# Import library
import pandas as pd  # Digunakan untuk mengolah data dalam bentuk tabel
import pickle  # Dipakai untuk menyimpan model yang sudah dilatih
from sklearn.tree import DecisionTreeClassifier  # Algoritma pohon keputusan untuk membuat model
from sklearn.model_selection import train_test_split  # Untuk membagi data menjadi training dan testing
from sklearn.preprocessing import LabelEncoder  # Untuk mengubah data kategori menjadi angka
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree
from matplotlib import pyplot as plt



# Load dataset
# Membaca file CSV yang berisi data penyakit jantung dan menyimpannya dalam dataframe
df = pd.read_csv('heart.csv')

# Preprocessing
# Mengubah kolom 'Sex' dari data kategori (Male/Female) menjadi angka (0/1)
kode_encoder = LabelEncoder()
df['Sex'] = kode_encoder.fit_transform(df['Sex'])

# Mengubah kolom kategori lainnya menjadi dummy variables (kolom baru berupa 0/1)
# Contohnya: 'ChestPainType' dipecah menjadi beberapa kolom seperti 'ChestPainType_ATYPICAL', 'ChestPainType_NONANGINAL', dll.
df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

# Memisahkan fitur dan target
# 'X' adalah semua kolom kecuali 'HeartDisease' (fitur yang digunakan untuk prediksi)
# 'y' adalah kolom 'HeartDisease' (target yang ingin diprediksi)
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Membagi data menjadi training dan testing
# Data dibagi menjadi 90% untuk training (latihan) dan 10% untuk testing (pengujian)
# random_state=50 memastikan hasil pembagian data selalu sama setiap kali kode dijalankan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)

# Membuat dan melatih model
# Menggunakan algoritma Decision Tree untuk membuat model prediksi
# Model dilatih dengan data training (X_train dan y_train)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Akurasi: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")

# Menyimpan model ke file .sav
# Model yang sudah dilatih disimpan dalam file dengan nama 'model_prediksi_gagal_jantung.sav'
# Tujuannya agar model bisa digunakan lagi tanpa perlu melatih ulang
with open('model_prediksi_gagal_jantung.sav', 'wb') as file:
    pickle.dump(model, file)

print("Model berhasil disimpan sebagai model_prediksi_gagal_jantung.sav")

plt.figure(figsize=(30,24))
tree.plot_tree(model, fontsize=16, rounded=True, filled= True)

plt.savefig("decision_tree_plot.jpg", format="jpg", dpi=300)  # Resolusi lebih tinggi
print("Gambar berhasil disimpan sebagai 'decision_tree_plot.jpg'")
