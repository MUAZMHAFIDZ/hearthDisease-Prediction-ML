# Import library
import pandas as pd  # Untuk pengolahan data dalam bentuk tabel
import streamlit as st  # Untuk membuat antarmuka web yang interaktif
# import matplotlib.pyplot as plt  # Untuk membuat grafik visualisasi
from matplotlib import pyplot as plt
import pickle  # Untuk memuat model yang sudah dilatih sebelumnya
from sklearn.tree import plot_tree  # Untuk menggambar pohon keputusan
from sklearn.preprocessing import LabelEncoder  # Untuk mengonversi data kategori menjadi angka

# Load image
# Membaca file gambar yang akan ditampilkan di halaman beranda
icon = plt.imread('heart-attack.png')
img = plt.imread('jantungg.jpg')

# Load dataset
# Membaca dataset penyakit jantung dari file CSV
df = pd.read_csv('heart.csv')

# Load pre-trained model
# Membuka model yang sudah dilatih sebelumnya dari file .sav
model = pickle.load(open('model_prediksi_gagal_jantung.sav', 'rb'))

# Streamlit interface
# Membuat judul aplikasi web
# st.title("Ayo kita prediksi Penyakit Jantung!")
st.markdown(
    """
    <h1 style="text-align: center; color: black;">Aplikasi Prediksi Penyakit Jantung</h1>
    """,
    unsafe_allow_html=True
)

# Left sidebar
# Sidebar untuk navigasi menu
st.sidebar.image(icon, width=100)
menu = st.sidebar.selectbox("Pilih Konten", ['Beranda', 'Dataset', 'Grafik', 'Prediksi'])

if menu == 'Beranda':
    st.image(img, caption='Gambar Jantung', use_container_width=True)
    
    # Menambahkan selamat datang dan penjelasan aplikasi
    st.markdown("""
    # **Selamat Datang di Aplikasi Prediksi Penyakit Jantung!**
    🔬 **Aplikasi ini menggunakan teknologi Machine Learning untuk memprediksi risiko penyakit jantung berdasarkan data kesehatan Anda.**
    
    ### 🤖 Teknologi yang Digunakan:
    - **Decision Tree**: Algoritma pohon keputusan yang digunakan untuk menentukan apakah seseorang berisiko terkena penyakit jantung berdasarkan fitur yang ada.
    - **Alasan**:    
        - **Mudah Dipahami**: Hasil prediksi berupa aturan yang jelas, memudahkan interpretasi oleh tenaga medis.
        - **Cocok untuk Data Kesehatan**: Mampu menangani data numerik dan kategorikal, seperti tekanan darah atau jenis nyeri dada.
        - **Efisien**: Cepat dalam memproses data untuk menghasilkan prediksi.
        - **Akurasi Baik**: Memberikan hasil yang akurat meskipun dataset terbatas.
        - **Transparan**: Proses pengambilan keputusan dapat dilihat dan dijelaskan secara logis.

    ---
    ## ✅ **Alasan Memilih Dataset Ini:**
    - **Kualitas Data**: Data lengkap dan akurat, sehingga hasil analisis dapat diandalkan.
    - **Kemudahan Penggunaan**: Format dataset mudah diakses dan dipahami, mempermudah analisis.
    - **Ukuran yang Tepat**: Dataset cukup besar untuk temuan signifikan, namun tidak terlalu besar untuk dikelola.
    - **Sumber Terpercaya**: Dataset berasal dari sumber yang kredibel, memastikan validitas data.

    ---
    ## 📋 **Fitur Utama:**
    - 📊 **Dataset**: Menyediakan data lengkap tentang penyakit jantung yang dapat dieksplorasi.
    - 📈 **Grafik Visualisasi**: Menyediakan grafik interaktif untuk analisis data.
    - 🔮 **Prediksi**: Melakukan prediksi penyakit jantung berdasarkan data input yang diberikan.

    ---
    ## 📝 **Dataset dan Sumber Data:**
    - Data berasal dari **[Kaggle-PrediksiPenyakitJantung](https://www.kaggle.com/datasets/anthonyrlam/heart-failure-prediction-ebm/data?select=utils_cardio.py)**, yang menyediakan data penyakit jantung yang lengkap dan terstruktur.

    ---
    ## 📑 **Cara Menggunakan Website:**
    1. Pilih menu "Dataset" untuk melihat data penyakit jantung.
    2. Pilih menu "Grafik" untuk melihat visualisasi data.
    3. Pilih menu "Prediksi" untuk memprediksi risiko penyakit jantung berdasarkan input Anda.

    **Siap untuk mulai?** Pilih menu di sebelah kiri dan jelajahi aplikasi ini!
    """)

# Menu Dataset
elif menu == 'Dataset':
    st.subheader("Dataset Penyakit Jantung")
    # Menampilkan seluruh dataset
    st.write(df)

    st.subheader("Analisis Deskriptif")
    # Menampilkan ringkasan statistik dataset
    st.write(df.describe())
    # Menampilkan jumlah masing-masing jenis nyeri dada
    st.write(df['ChestPainType'].value_counts())

# Menu Grafik
elif menu == 'Grafik':
    st.subheader("Visualisasi Data")
    
    # Histogram distribusi usia
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(df['Age'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Usia', fontsize=14)
    ax.set_ylabel('Jumlah', fontsize=14)
    ax.set_title('Distribusi Usia', fontsize=16, fontweight='bold')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    st.pyplot(fig)

    # Scatter plot usia vs kolesterol
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        df['Age'], df['Cholesterol'], 
        c=df['Cholesterol'], cmap='viridis', alpha=0.7, edgecolor='k'
    )
    ax.set_xlabel('Usia', fontsize=14)
    ax.set_ylabel('Kolesterol', fontsize=14)
    ax.set_title('Hubungan Usia dan Kolesterol', fontsize=16, fontweight='bold')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Tambahkan colorbar untuk menunjukkan intensitas kolesterol
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Level Kolesterol', fontsize=12)
    st.pyplot(fig)

    # diagram pengidap penyakit jantung tiap usia
    plt.figure(figsize=(8, 4))
    plt.plot(df['Age'], df['HeartDisease'], label="Penyakit Jantung Di Tiap Usia")
    plt.title("Penyakit Jantung Di Tiap Usia")
    plt.xlabel("Usia")
    plt.ylabel("Penyakit Jantung")
    plt.legend()
    st.pyplot(plt)
    

# Menu Prediksi
elif menu == 'Prediksi':
    st.subheader("Prediksi Penyakit Jantung dengan Decision Tree")

    # Preprocessing dataset
    # Encode kolom 'Sex' dan ubah kategori lain menjadi dummy variables
    kode_encoder = LabelEncoder()
    df['Sex'] = kode_encoder.fit_transform(df['Sex'])
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

    # Memisahkan fitur dan target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    st.header("Input Data Baru")
    col1, col2 = st.columns([3, 1])
    # Form input data baru di Streamlit
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=40)
        sex = st.selectbox("Sex", options=['M', 'F'])
        chest_pain_type = st.selectbox("Chest Pain Type", options=['ATA', 'NAP', 'ASY', 'TA'])
        resting_bp = st.number_input("RestingBP", min_value=50, max_value=200, value=120)
        cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
        fasting_bs = st.selectbox("FastingBS", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        resting_ecg = st.selectbox("RestingECG", options=['Normal', 'ST', 'LVH'])
        max_hr = st.number_input("MaxHR", min_value=50, max_value=250, value=150)
        exercise_angina = st.selectbox("Exercise Angina", options=['Y', 'N'])
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
        st_slope = st.selectbox("ST_Slope", options=['Up', 'Flat', 'Down'])

    # Data baru diubah ke dalam bentuk DataFrame
    data_baru = {
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain_type],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    }
    df_baru = pd.DataFrame(data_baru)

    # Preprocess data baru
    df_baru['Sex'] = kode_encoder.transform(df_baru['Sex'])
    df_baru = pd.get_dummies(df_baru, columns=['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
    df_baru = df_baru.reindex(columns=X.columns, fill_value=0)

    # Prediksi
    with col2:
        if st.button("Prediksi"):
            prediksi = model.predict(df_baru)
            if prediksi[0] == 1:
                iconic = plt.imread('medical.png')
                hasil = "Pasien mengalami Risiko Penyakit Jantung"
                color = "#8B0000"  # Merah untuk risiko tinggi
            else:
                iconic = plt.imread('healthy.png')
                hasil = "Pasien tidak Mengalami Risiko Penyakit Jantung"
                color = "green"  # Hijau untuk tidak ada risiko

            # Menampilkan hasil dengan warna yang sesuai
            st.image(iconic, width=150)
            st.markdown(f'<h4 style="color:{color};">{hasil}</h4>', unsafe_allow_html=True)