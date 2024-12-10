# Import library
import pandas as pd  # Untuk mengolah data dalam bentuk tabel
import pickle  # Untuk menyimpan model yang sudah dilatih
from sklearn.tree import DecisionTreeClassifier  # Algoritma pohon keputusan
from sklearn.ensemble import RandomForestClassifier  # Algoritma random forest
from sklearn.model_selection import train_test_split  # Membagi data menjadi training dan testing
from sklearn.preprocessing import LabelEncoder  # Mengubah data kategori menjadi angka
from sklearn.metrics import accuracy_score, classification_report  # Evaluasi model

# Load dataset
df = pd.read_csv('heart.csv')  # Membaca dataset

# Preprocessing
# Mengubah kolom 'Sex' dari data kategori (Male/Female) menjadi angka (0/1)
kode_encoder = LabelEncoder()
df['Sex'] = kode_encoder.fit_transform(df['Sex'])

# Mengubah kolom kategori lainnya menjadi dummy variables
df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

# Memisahkan fitur dan target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)

# ===================== Decision Tree =====================
# Membuat dan melatih model Decision Tree
dt_model = DecisionTreeClassifier(random_state=50)
dt_model.fit(X_train, y_train)

# Menyimpan model Decision Tree ke file
with open('model_decision_tree.sav', 'wb') as file:
    pickle.dump(dt_model, file)

# Evaluasi Decision Tree
dt_predictions = dt_model.predict(X_test)
print("Decision Tree Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, dt_predictions):.2f}")
print(classification_report(y_test, dt_predictions))

# ===================== Random Forest =====================
# Membuat dan melatih model Random Forest
rf_model = RandomForestClassifier(random_state=50, n_estimators=100)
rf_model.fit(X_train, y_train)

# Menyimpan model Random Forest ke file
with open('model_random_forest.sav', 'wb') as file:
    pickle.dump(rf_model, file)

# Evaluasi Random Forest
rf_predictions = rf_model.predict(X_test)
print("\nRandom Forest Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, rf_predictions):.2f}")
print(classification_report(y_test, rf_predictions))
