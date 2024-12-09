import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

# 1. Membaca Data CSV
file_path = "heart_2020_cleaned.csv"  # Ganti dengan nama file CSV Anda
data = pd.read_csv(file_path)

# 2. Mengonversi kolom target (ResikoJantung) menjadi numerik
data["ResikoJantung"] = data["ResikoJantung"].map({"No": 0, "Yes": 1})

# 3. Memisahkan Fitur dan Target
X = data.drop("ResikoJantung", axis=1)
y = data["ResikoJantung"]

# 4. Mengonversi fitur kategorikal ke numerik menggunakan One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# 5. Membagi Data (Train-Test Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Membuat dan Melatih Model Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Evaluasi Model
y_pred = model.predict(X_test)
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))
print("Akurasi model pada data uji:", accuracy_score(y_test, y_pred))

# 8. Menyimpan Model dan Kolom
joblib.dump(model, "model_risiko_jantung.pkl")
joblib.dump(X.columns, "fit_columns.pkl")

# 9. Fungsi untuk Konversi Input "iya" dan "tidak" ke 1 dan 0
def konversi_input(jawaban):
    if jawaban.lower() in ["iya", "ya"]:
        return 1
    elif jawaban.lower() == "tidak":
        return 0
    else:
        raise ValueError("Masukkan hanya 'iya' atau 'tidak'.")

# 10. Fungsi untuk Prediksi Berdasarkan Input User
def prediksi_risiko():
    print("\nMasukkan data Anda untuk memprediksi risiko jantung:")
    try:
        # Input pengguna
        bmi = float(input("BMI (contoh: 25.3): "))
        merokok = konversi_input(input("Apakah Anda merokok? (iya/tidak): "))
        alkohol = konversi_input(input("Apakah Anda mengonsumsi alkohol? (iya/tidak): "))
        stroke = konversi_input(input("Apakah Anda memiliki riwayat stroke? (iya/tidak): "))
        mental = float(input("Berapa hari dalam sebulan Anda merasa tidak sehat secara mental? (contoh: 5): "))
        diffwalking = konversi_input(input("Apakah Anda memiliki kesulitan berjalan? (iya/tidak): "))
        jenis_kelamin = input("Jenis kelamin Anda (Laki-laki/Perempuan): ").lower()
        gender = 1 if jenis_kelamin == "laki-laki" else 0
        umur = input("Kategori umur Anda (contoh: 55-59): ")
        aktif = konversi_input(input("Apakah Anda aktif secara fisik? (iya/tidak): "))
        tidur = float(input("Berapa jam tidur Anda rata-rata setiap malam? (contoh: 6.5): "))

        # Menyusun data input
        user_data = {
            "BMI": bmi,
            "Smoking_Yes": merokok,
            "AlcoholDrinking_Yes": alkohol,
            "Stroke_Yes": stroke,
            "MentalHealth": mental,
            "DiffWalking_Yes": diffwalking,
            "Sex_Male": gender,
            "AgeCategory_" + umur: 1,
            "PhysicalActivity_Yes": aktif,
            "SleepTime": tidur,
        }

        # Memuat model dan kolom
        loaded_model = joblib.load("model_risiko_jantung.pkl")
        fit_columns = joblib.load("fit_columns.pkl")

        # Membuat DataFrame input yang sesuai dengan fitur pelatihan
        user_df = pd.DataFrame([user_data], columns=fit_columns).fillna(0)
        print("\nData input user:\n", user_df)  # Debug data input

        # Prediksi Risiko
        hasil = loaded_model.predict(user_df)
        risiko = "Risiko Tinggi" if hasil[0] == 1 else "Risiko Rendah"
        print(f"\nPrediksi Risiko Jantung Anda: {risiko}")

        # Membuat diagram untuk hasil prediksi
        plt.figure(figsize=(6, 4))

        labels = ['Risiko Tinggi', 'Risiko Rendah']
        counts = [1 if hasil[0] == 1 else 0, 1 if hasil[0] == 0 else 0]
        colors = ['red', 'green']

        # Membuat bar chart
        bars = plt.bar(labels, counts, color=colors)

        # Menambahkan anotasi pada bar
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    str(count), ha='center', va='center', fontsize=12, color='white', fontweight='bold')

        # Menambahkan judul dan label sumbu
        plt.title("Prediksi Risiko Jantung", fontsize=14, fontweight='bold')
        plt.ylabel("Jumlah", fontsize=12)
        plt.xlabel("Kategori Risiko", fontsize=12)

        # Menampilkan grid untuk estetika
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Menampilkan plot
        plt.tight_layout()
        plt.show()

    except ValueError as e:
        print(f"Error: {e}. Coba lagi dengan masukan yang benar.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# 11. Memulai Prediksi
prediksi_risiko()