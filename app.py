# Import library
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from matplotlib import pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from graphviz import Source
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load image
icon = plt.imread('heart-attack.png')
img = plt.imread('jantungg.jpg')

# Load dataset
df = pd.read_csv('heart.csv')

# Load pre-trained model
model = pickle.load(open('model_prediksi_gagal_jantung.sav', 'rb'))

# Sidebar interaktif dengan option_menu
with st.sidebar:
    selected = option_menu(
        menu_title="Pilih Menu",
        options=["Beranda", "Dataset", "Grafik", "Prediksi", "Tentang Kami"],
        icons=["house", "table", "bar-chart", "activity", "person-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f8f9fa"},
            "icon": {"color": "#E195AB", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#e8f4fd",
            },
            "nav-link-selected": {"background-color": "#87A2FF", "color": "white"},
            "menu-title": {"color": "black", "font-size": "16px", "text-align": "center", "margin-bottom": "10px"},
        },
    )

# Menu Beranda
if selected == "Beranda":
    st.markdown("""
    # **Selamat Datang di Aplikasi Prediksi Penyakit Jantung!**
    🔬 **Aplikasi ini menggunakan teknologi Machine Learning untuk memprediksi risiko penyakit jantung berdasarkan data kesehatan Anda.** """)
    st.image(img, caption='Gambar Jantung', use_container_width=True)
    (""" ### 🤖 Teknologi yang Digunakan:
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
elif selected == "Dataset":
    st.subheader("Dataset Penyakit Jantung")
    st.write(df)
    st.subheader("Analisis Deskriptif")
    st.write(df.describe())
    st.write(df['ChestPainType'].value_counts())

# Menu Grafik
elif selected == "Grafik":
    st.subheader("Visualisasi Data")
    # Histogram distribusi usia
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(df['Age'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Usia', fontsize=14)
    ax.set_ylabel('Jumlah', fontsize=14)
    ax.set_title('Distribusi Usia', fontsize=16, fontweight='bold')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    st.pyplot(fig)

# Menu Prediksi
elif selected == 'Prediksi':
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
    st.header("Input Data Baru")
    age = st.number_input("Age || Umur ", min_value=1, max_value=120, help="Umurmu brp?")
    sex = st.selectbox("Sex || Jenis Kelamin", options=['M', 'F'], help="Hanya menerima 2 gender")
    chest_pain_type = st.selectbox("Chest Pain Type || Tipe Nyeri Dada", options=['ATA (Atypical Angina)', 'NAP (Non-Anginal Pain)', 'ASY (Asymptomatic)', 'TA (Typical Angina)'])
    resting_bp = st.number_input("Resting Blood Pressure || Tekanan darah saat istirahat ", min_value=50, max_value=200, help="min 50, max : 200")
    cholesterol = st.number_input("Cholesterol || Kolesterol", min_value=100, max_value=600, help="min 100, max 600")
    fasting_bs = st.selectbox("Fasting Blood Sugar || Gula Darah Puasa", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    resting_ecg = st.selectbox("Resting Electrocardiographic || Elektrokardiograpik saat istirahat", options=['Normal', 'ST', 'LVH'], help="Hasil elektrokardiogram (EKG) yang diambil saat pasien dalam keadaan istirahat")
    max_hr = st.number_input("Max Heart Rate || Denyut Jantung Maksimum", min_value=50, max_value=250, help="Denyut jantung tertinggi yang tercatat saat pasien beraktivitas" )
    exercise_angina = st.selectbox("Exercise Angina || Angina yang Dihasilkan oleh Olahraga", options=['Y', 'N'],help=" Apakah pasien mengalami nyeri dada saat berolahraga")
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, help="Mengukur perubahan segmen ST setelah beraktivitas (lebih banyak penurunan menunjukkan lebih tinggi risiko masalah jantung)")
    st_slope = st.selectbox("ST_Slope", options=['Up', 'Flat', 'Down'], help= "Mengukur kemiringan segmen ST setelah beraktivitas (downsloping menunjukkan kemungkinan penyakit jantung yang lebih serius, upslope biasanya normal).")

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

    # Visualisasi pohon keputusan
    if st.checkbox("Tampilkan Struktur Pohon Keputusan"):
        tree_rules = export_graphviz(
            model, 
            out_file=None, 
            feature_names=X.columns, 
            class_names=['Tidak Berisiko', 'Berisiko'], 
            filled=True, 
            rounded=True, 
            special_characters=True
        )
        st.graphviz_chart(tree_rules)

# Menu About Us
elif selected == "Tentang Kami":
    st.subheader("Tentang Kami")
    st.markdown("""
    ## **Tim Pengembang**
    Kami adalah tim yang berdedikasi dalam mengembangkan aplikasi berbasis machine learning untuk memprediksi risiko penyakit jantung.
    ### **Kontak Kami**
    - 📧 Email: support@prediksijantung.com
    """)
