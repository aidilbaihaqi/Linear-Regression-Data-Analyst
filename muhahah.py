import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Membaca data dari file CSV
def read_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data berhasil dibaca:")
        print(data.head())
        return data
    except Exception as e:
        print(f"Terjadi kesalahan saat membaca file: {e}")
        return None

# Melakukan pemrosesan data dan regresi linear
def perform_regression(data):
    try:
        # Memisahkan fitur (X1-X6) dan target (Y)
        X = data[["X1", "X2", "X3", "X4", "X5", "X6"]]
        y = data["Y"]

        # Membagi data menjadi data latih dan data uji
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Membuat model regresi linear
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Melakukan prediksi pada data uji
        y_pred = model.predict(X_test)

        # Menampilkan hasil
        print("Koefisien regresi:", model.coef_)
        print("Intercept:", model.intercept_)
        print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
        print("R^2 Score:", r2_score(y_test, y_pred))

        return model

    except Exception as e:
        print(f"Terjadi kesalahan saat melakukan regresi: {e}")
        return None

if __name__ == "__main__":
    # Path ke file CSV
    file_path = "data_pengeluaran_mahasiswa.csv"

    # Membaca data
    data = read_data(file_path)

    if data is not None:
        # Memastikan kolom yang diperlukan ada
        required_columns = ["X1", "X2", "X3", "X4", "X5", "X6", "Y"]
        if all(col in data.columns for col in required_columns):
            model = perform_regression(data)

            if model:
                print("Regresi selesai.")
            else:
                print("Gagal melakukan regresi.")
        else:
            print("Data tidak memiliki kolom yang diperlukan:", required_columns)
    else:
        print("Gagal membaca data.")
