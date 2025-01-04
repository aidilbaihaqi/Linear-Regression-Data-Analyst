import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Membaca data dari file CSV
def read_data(file_path):
    try:
        data = pd.read_csv(file_path)
        messagebox.showinfo("Info", "Data berhasil dibaca.")
        return data
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan saat membaca file: {e}")
        return None

# Melakukan pemrosesan data dan regresi linear
def perform_regression(data):
    try:
        X = data[["X1", "X2", "X3", "X4", "X5", "X6"]]
        y = data["Y"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.grid(True)
        plt.figtext(0.15, 0.02, f"Koefisien: {np.round(model.coef_, 2)}\nIntercept: {np.round(model.intercept_, 2)}\nMSE: {np.round(mean_squared_error(y_test, y_pred), 2)}\nR^2 Score: {np.round(r2_score(y_test, y_pred), 2)}", fontsize=10, ha="left")
        plt.show()

        kesimpulan = ""
        if model.intercept_ > 0:
            kesimpulan += "Intercept positif menunjukkan adanya nilai tabungan awal yang tidak dipengaruhi oleh variabel independen.\n"
        else:
            kesimpulan += "Intercept negatif menunjukkan potensi defisit meskipun tidak ada pengaruh dari variabel independen.\n"

        for i, coef in enumerate(model.coef_):
            if coef > 0:
                kesimpulan += f"Variabel X{i+1} memiliki pengaruh positif terhadap tabungan. Semakin tinggi nilai X{i+1}, semakin tinggi nilai Y.\n"
            else:
                kesimpulan += f"Variabel X{i+1} memiliki pengaruh negatif terhadap tabungan. Semakin tinggi nilai X{i+1}, semakin rendah nilai Y.\n"

        messagebox.showinfo("Kesimpulan", kesimpulan)

        return model

    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan saat melakukan regresi: {e}")
        return None

# Fungsi utama untuk menjalankan UI
def main():
    def open_file():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            data = read_data(file_path)
            if data is not None:
                required_columns = ["X1", "X2", "X3", "X4", "X5", "X6", "Y"]
                if all(col in data.columns for col in required_columns):
                    perform_regression(data)
                else:
                    messagebox.showwarning("Peringatan", "Data tidak memiliki kolom yang diperlukan: " + ", ".join(required_columns))

    # Membuat UI
    root = tk.Tk()
    root.title("Regresi Linear Berganda")

    label = tk.Label(root, text="Pilih file CSV untuk melakukan regresi linear.")
    label.pack(pady=20)

    button = tk.Button(root, text="Pilih File", command=open_file)
    button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    # Path ke file CSV
    file_path = "data.csv"

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
