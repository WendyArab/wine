import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier  # Mengganti RandomForestClassifier dengan ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st

# Ganti 'wine_dataset.csv' dengan nama file dataset yang sesuai
url = "http://archive.ics.uci.edu/static/public/109/data.csv"
wine_data = pd.read_csv(url)

# Pisahkan atribut dan label
X = wine_data.drop('class', axis=1)
y = wine_data['class']

# Pisahkan dataset menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model Extra Trees Classifier
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)  # Mengganti RandomForestClassifier dengan ExtraTreesClassifier

# Latih model pada data pelatihan
et_model.fit(X_train, y_train)

# Fungsi untuk memprediksi kelas anggur
def predict_wine_class(features):
    prediction = et_model.predict([features])
    return prediction[0]

# Judul aplikasi
st.title("Aplikasi Klasifikasi Jenis Anggur dengan Extra Trees Classifier")  # Mengganti judul aplikasi

# Tampilkan akurasi model
y_pred = et_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Akurasi Model: {accuracy:.2f}")

# Memungkinkan pengguna memasukkan semua fitur
input_features = []
for feature_name in wine_data.columns[:-1]:  # Menghindari kolom 'class'
    value = st.slider(f"Masukkan nilai untuk {feature_name}", min_value=min(wine_data[feature_name]), max_value=max(wine_data[feature_name]))
    input_features.append(value)

# Tombol untuk melakukan prediksi
if st.button("Prediksi Jenis Anggur"):
    if len(input_features) == len(wine_data.columns) - 1:  # Memastikan ada 13 nilai input
        prediction = predict_wine_class(input_features)
        st.write(f"Jenis Anggur yang Diprediksi: Kelas {prediction}")
    else:
        st.write("Harap masukkan nilai untuk semua fitur")

# Tampilkan dataset jika diinginkan
if st.checkbox("Tampilkan Dataset"):
    st.write(wine_data)
