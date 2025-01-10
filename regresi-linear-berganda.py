import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm

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

        # Menambahkan konstanta untuk intersep
        X = sm.add_constant(X)

        # Melakukan regresi linear menggunakan statsmodels
        model = sm.OLS(y, X).fit()

        # Menampilkan hasil regresi
        print(model.summary())

        # Membagi data untuk visualisasi
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.grid(True)
        plt.figtext(0.15, 0.02, f"MSE: {np.round(mean_squared_error(y_test, y_pred), 2)}\nR^2 Score: {np.round(r2_score(y_test, y_pred), 2)}", fontsize=10, ha="left")
        plt.show()

        # Kesimpulan berdasarkan hasil
        kesimpulan = f"Hasil regresi menunjukkan bahwa R^2 = {model.rsquared:.2f}.\n"
        for i, coef in enumerate(model.params[1:], start=1):  # Skip const
            pval = model.pvalues[i]
            signifikansi = "signifikan" if pval < 0.05 else "tidak signifikan"
            kesimpulan += f"Variabel X{i} memiliki koefisien {coef:.2f} dan {signifikansi} dengan p-value {pval:.4f}.\n"

        print(kesimpulan)
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
