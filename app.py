# Import library
import pandas as pd  # Untuk pengolahan data dalam bentuk tabel
import streamlit as st  # Untuk membuat antarmuka web yang interaktif
#import matplotlib.pyplot as plt  # Untuk membuat grafik visualisasi
from matplotlib import pyplot as plt
import pickle  # Untuk memuat model yang sudah dilatih sebelumnya
from sklearn.tree import plot_tree  # Untuk menggambar pohon keputusan
from sklearn.preprocessing import LabelEncoder  # Untuk mengonversi data kategori menjadi angka

# Load image
# Membaca file gambar yang akan ditampilkan di halaman beranda
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
    <h1 style="text-align: center; color: black;">Data Penyakit Jantung</h1>
    """,
    unsafe_allow_html=True
)

# Left sidebar
# Sidebar untuk navigasi menu
menu = st.sidebar.selectbox("Pilih Konten", ['Beranda', 'Dataset', 'Grafik', 'Prediksi'])

# Menu Beranda
if menu == 'Beranda':
    st.image(img, caption='Gambar Jantung', use_container_width=True)
    st.markdown("""
    #  **Selamat Datang di Data Penyakit Jantung**!  
    🔬 **Mari jelajahi data, temukan wawasan, dan tingkatkan kesadaran Anda tentang kesehatan jantung.**  
    🌟 Aplikasi ini dirancang untuk memberikan Anda pengalaman eksplorasi yang informatif dan menarik.

    ---
    ## 📋 **Apa yang Bisa Anda Lakukan di Sini?**
    - 🔍 **Data Set** data penyakit jantung dengan visual yang interaktif.
    - 📊 **Grafik** tren kesehatan untuk wawasan yang lebih dalam.
    - 📚 **Prediksi** penting untuk hidup lebih sehat!

    ---
    🚀 **Siap Memulai?**  
    Pilih menu di **sebelah kiri** dan mulailah perjalanan Anda untuk memahami lebih jauh tentang kesehatan jantung!  
    """)
    st.success("🌟 Hidup Sehat Dimulai dari Langkah Kecil Hari Ini!")
    st.write("Selamat datang di web Kami!")
    st.write("Aplikasi ini memungkinkan Anda untuk menjelajahi data terkait penyakit jantung.")
    st.write("Silakan pilih menu di sebelah kiri untuk melihat konten yang tersedia.")

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
    st.subheader("Prediksi Penyakit Jantung")

    # Preprocessing dataset
    # Encode kolom 'Sex' dan ubah kategori lain menjadi dummy variables
    kode_encoder = LabelEncoder()
    df['Sex'] = kode_encoder.fit_transform(df['Sex'])
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

    # Memisahkan fitur dan target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # Form input data baru di Streamlit
    st.title("Prediksi Penyakit Jantung dengan Decision Tree")
    st.header("Input Data Baru")
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
    if st.button("Prediksi"):
        prediksi = model.predict(df_baru)
        hasil = "Pasien mengalami Risiko Penyakit Jantung" if prediksi[0] == 1 else "Pasien tidak Mengalami Risiko Penyakit Jantung"
        st.subheader(f"Hasil Prediksi: {hasil}")
