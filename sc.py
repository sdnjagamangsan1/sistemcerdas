import streamlit as st
import numpy as np
import cv2
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

# Fungsi untuk memuat dan memproses gambar
def load_and_process_image(image):
    image = np.array(image.convert('L'))  # Konversi gambar ke grayscale
    resized_image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    flattened_image = resized_image.flatten()
    input_data = flattened_image.reshape(1, -1)
    return input_data

# Fungsi untuk melatih model
def train_model():
    # Muat dataset digits (angka tulisan tangan)
    digits = datasets.load_digits()
    data = digits.data
    target = digits.target

    # Bagi dataset menjadi data pelatihan (80%) dan data pengujian (20%)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Normalisasi data menggunakan StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Bangun model jaringan saraf tiruan
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    
    # Latih model dengan data pelatihan
    model.fit(X_train, y_train)
    
    return model, scaler, X_test, y_test

# Judul aplikasi
st.title('Aplikasi Sistem Cerdas Ida Hafizah')

# Bagian 1: Contoh input dan output
st.write('Selamat datang di aplikasi praktikum berbasis Streamlit!')

name = st.text_input('Masukkan nama Anda:')
if name:
    st.write(f'Halo, {name}!')

# Bagian 2: Klasifikasi Angka Tulisan Tangan
st.write('## Klasifikasi Angka Tulisan Tangan')
st.write('Unggah gambar angka tulisan tangan Anda untuk diklasifikasikan.')

# Unggah gambar
uploaded_file = st.file_uploader("Unggah gambar angka tulisan tangan", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    # Muat dan proses gambar
    input_data = load_and_process_image(image)
    
    # Latih model
    model, scaler, X_test, y_test = train_model()
    
    # Transform input data
    input_data = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_data)
    st.write(f'Prediksi: {prediction[0]}')
    
    # Evaluasi model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
    st.write(f'Akurasi Model: {accuracy}')
    st.write('Classification Report:')
    st.json(classification_rep)
else:
    st.write("Silakan unggah file gambar dalam format PNG, JPG, atau JPEG.")
