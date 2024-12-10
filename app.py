# Import library
import pandas as pd  # Untuk pengolahan data dalam bentuk tabel
import streamlit as st  # Untuk membuat antarmuka web yang interaktif
from streamlit_option_menu import option_menu  # Untuk sidebar dengan desain interaktif
from matplotlib import pyplot as plt  # Untuk membuat grafik visualisasi
import pickle  # Untuk memuat model yang sudah dilatih sebelumnya
from sklearn.preprocessing import LabelEncoder  # Untuk mengonversi data kategori menjadi angka
from sklearn.tree import export_graphviz  # Untuk menggambar pohon keputusan
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from graphviz import Source  # Untuk menampilkan struktur pohon keputusan

# Load image
icon = plt.imread('heart-attack.png')  # Ikon untuk tampilan prediksi
img = plt.imread('jantungg.jpg')  # Gambar jantung untuk beranda

# Load dataset
df = pd.read_csv('heart.csv')  # Dataset penyakit jantung

# Load pre-trained model
model = pickle.load(open('model_prediksi_gagal_jantung.sav', 'rb'))  # Model machine learning

# Sidebar interaktif dengan option_menu
with st.sidebar:
    selected = option_menu(
        menu_title="Pilih Menu",
        options=["Beranda", "Dataset", "Grafik", "Prediksi", "Cara Kerja", "Tentang Kami"],
        icons=["house", "table", "bar-chart", "activity", "gear", "person-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f8f9fa"},
            "icon": {"color": "#E195AB", "font-size": "16px"},
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
    🔬 **Aplikasi ini menggunakan teknologi Machine Learning untuk memprediksi risiko penyakit jantung berdasarkan data kesehatan Anda.**
    """)
    st.image(img, caption='Gambar Jantung', use_container_width=True)
    st.markdown("""
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

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Level Kolesterol', fontsize=12)
    st.pyplot(fig)

# Prediction Menu
elif selected == 'Prediksi':
    st.subheader("Heart Disease Prediction with Decision Tree")

    # Preprocess the dataset
    # Encode 'Sex' and create dummy variables for other categorical columns
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

    # Separate features and target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    st.header("Input Data Baru")

    # Input form
    age = st.number_input("Age || Umur", min_value=1, max_value=120, value=40, help="Umurmu brp?")
    sex = st.selectbox("Sex || Jenis Kelamin", options=['M', 'F'],help="Hanya menerima 2 gender" )
    chest_pain_type = st.selectbox("Chest Pain Type || Tipe Nyeri Dada", options=['ATA', 'NAP', 'ASY', 'TA'],help="ATA = Atypical Angina, 'NAP = Non-Anginal Pain, ASY = Asymptomatic, TA = Typical Angina" )
    resting_bp = st.number_input("Resting Blood Pressure || Tekanan darah saat istirahat (mmHg)", min_value=50, max_value=200, value=120, help="min 50, max : 200")
    cholesterol = st.number_input("Cholesterol || Kolesterol (mg/dL)", min_value=100, max_value=600, value=200, help="min 100, max 600")
    fasting_bs = st.selectbox("Fasting Blood Sugar || Gula Darah Puasa > 120 mg/dL", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    resting_ecg = st.selectbox("Resting Electrocardiographic || Elektrokardiograpik saat istirahat", options=['Normal', 'ST', 'LVH'],help="Hasil elektrokardiogram (EKG) yang diambil saat pasien dalam keadaan istirahat")
    max_hr = st.number_input("Max Heart Rate || Denyut Jantung Maksimum", min_value=50, max_value=250, value=150, help="Denyut jantung tertinggi yang tercatat saat pasien beraktivitas")
    exercise_angina = st.selectbox("Exercise Angina || Angina yang Dihasilkan oleh Olahraga", options=['Y', 'N'],help=" Apakah pasien mengalami nyeri dada saat berolahraga")
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, help="Mengukur perubahan segmen ST setelah beraktivitas (lebih banyak penurunan menunjukkan lebih tinggi risiko masalah jantung)")
    st_slope = st.selectbox("ST_Slope", options=['Up', 'Flat', 'Down'], help= "Mengukur kemiringan segmen ST setelah beraktivitas (downsloping menunjukkan kemungkinan penyakit jantung yang lebih serius, upslope biasanya normal).")

    # Create a DataFrame for the new data
    new_data = {
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
    new_df = pd.DataFrame(new_data)

    # Preprocess the new data
    new_df['Sex'] = label_encoder.transform(new_df['Sex'])
    new_df = pd.get_dummies(new_df, columns=['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
    new_df = new_df.reindex(columns=X.columns, fill_value=0)

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(new_df)
        if prediction[0] == 1:
            image = plt.imread('medical.png')
            result = "Pasien ada resiko gagal jantung"
            color = "#8B0000"  # Red for high risk
        else:
            image = plt.imread('healthy.png')
            result = "Pasien aman sehat sentosa"
            color = "green"  # Green for low risk

        # Display the result with appropriate color and image
        st.image(image, width=150)
        st.markdown(f'<h4 style="color:{color};">{result}</h4>', unsafe_allow_html=True)

            
    # Visualisasi pohon keputusan
    #if st.checkbox("Tampilkan Struktur Pohon Keputusan"):
    #   tree_rules = export_graphviz(
    #        model,
    #       out_file=None,
    #       feature_names=X.columns,
    #        class_names=['Tidak Berisiko', 'Berisiko'],
    #        filled=True,
    #        rounded=True,
    #        special_characters=True
    #    )
    #    st.graphviz_chart(tree_rules)

elif selected == 'Cara Kerja':
    st.header ("Cara Kerja Decision Tree")
    st.image ("decision_tree_plot.jpg", caption="ini if else nya decision tree", use_container_width=True)
    # # Define the path to the image
    # image_path = 'decision_tree_plot.jpg'

    # # Define the HTML hyperlink with the image
    # html_string = f'<a href="{image_path}" target="_blank"><img src="{image_path}" width="200" caption="ini if else nya decision tree"></a>'

    # # Display the image using st.markdown
    # st.markdown(html_string, unsafe_allow_html=True)

    st.info("Penjelasan")
    st.subheader("Rumus Gini Index")
    st.latex(r'''
    Gini = 1 - \sum_{i=1}^{n} \left( \frac{count_i}{total\_samples} \right)^2
    ''')

    # Teks yang akan ditampilkan di dalam kotak
    teks = """
    Ambil contoh tree level 1 bagian True, tertera seperti ini :
    x[7] <= 0.5
    gini = 0.291
    samples = 476
    value = [84, 392]
    
    Penjelasannya per baris adalah
    x[7] <= 0.5 artinya kolom ke 8 (mulai dari 0) yaitu MaxHR, lalu di ambil nilai tengahnya di 0.5
    gini = 0.291 artinya tingkat impurity (kotor) / purity (bersih), semakin mendekati angka 0 maka semakin bersih
    samples = 476, artinya data yang digunakan, namun sudah di filter pada level 0
    value = [84, 392] artinya dalam 476 sample sebanyak 84 data bilang tidak kena penyakit jantung (False), dan sebanyak 392 data kena gagal jantung (True)

    cara ngitung gini sesuai dengan rumus di atas
    diketahui samples nya 476, data sakitnya 84/476 dan data sehat 329/476
    hasil pembagian adalah 
    sakit : 84/476 = 0.1764705882352941
    sehat : 329/476 = 0.8235294117647059
    kemudian di pangkat 2 dapet hasilnya
    sakit = 0.03114186851211072041522491349481 +
    sehat = 0.67820069204152252041522491349481 = 0.70934256055363324083044982698962
    hasilnya 0.29065743944636675916955017301038
    bulatkan jadi 0.291
    jadi Gini = 0.29065743944636675916955017301038 atau 0.291



    pada level 0
    x[18] artinya mungkin ada penambahan kolom saat proses ml berlangsung seperti hot encoding yang di chest type 
    samples yang digunakan hanya 826 namun dataset ada 918 hal ini memang di karenakan test_size di model.py yang menunjukan 10%
    918 di kali 10/100 hasilnya 91,8 atau 92, jadi data yang digunakan untuk pengetesan adalah 92 sedangkan data untuk training 
    yaitu 918 - 92 = 826 yaitu samples pada level 0
    """

    # Membuat kotak seperti textarea
    st.markdown(
        f"""
        <div style="
            border: 1px solid gray;
            padding: 10px;
            border-radius: 5px;
            background-color: #f9f9f9;
            color: black;
            font-family: monospace;
            white-space: pre-wrap; /* Supaya line break terlihat */
            overflow-x: auto; /* Untuk teks panjang agar bisa di-scroll horizontal */
        ">
            {teks}
        
        """,
        unsafe_allow_html=True,
    )

# Menu Tentang Kami
elif selected == "Tentang Kami":
    st.subheader("Tentang Kami")
    st.markdown("""
    ## **Tim Pengembang**
    Halo! Kami adalah tim mahasiswa semester 3 yang punya passion besar di dunia teknologi, khususnya machine learning. Lewat aplikasi ini, kami ingin membantu orang-orang buat lebih sadar soal kesehatan jantung mereka. Walaupun masih belajar, kami yakin usaha kecil ini bisa memberi dampak besar ke masyarakat.    
    ### **Kontak Kami**
    - Annisya / 233307094 
    - Haqqi / 233307115
    - Muaz / 233307107
    - Nihlatansya / 233307110
    """)
