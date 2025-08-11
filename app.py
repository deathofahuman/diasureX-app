import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox, ttk
import time
import threading

splash_root = tk.Tk()
splash_root.overrideredirect(True)
splash_root.geometry("500x300+500+250")
splash_root.configure(bg="#1B1B1B")  

splash_label = tk.Label(
    splash_root,
    text="🚀 Launching DiaSureX...",
    font=("Orbitron", 20, "bold"),
    bg="#1B1B1B",
    fg="#00E5FF"  
)
splash_label.pack(expand=True)

def load_main_app():
    time.sleep(2)
    splash_root.destroy()

threading.Thread(target=load_main_app).start()
splash_root.mainloop()

root = tk.Tk()
root.title("DiaSureX - Futuristic Diamond Tester")
root.geometry("500x650")
root.configure(bg="#F1F8E9")  
root.resizable(False, False)

try:
    df = pd.read_excel("diamond_training_data.xlsx")

    df['UV'] = df['UV'].fillna("None")
    df['Phosphorescence'] = df['Phosphorescence'].fillna("None")

    uv_encoder = LabelEncoder()
    phos_encoder = LabelEncoder()
    label_encoder = LabelEncoder()

    uv_encoder.fit(df['UV'])
    phos_encoder.fit(df['Phosphorescence'])
    label_encoder.fit(df['Label'])

    df_encoded = df.copy()
    df_encoded['UV'] = uv_encoder.transform(df_encoded['UV'])
    df_encoded['Phosphorescence'] = phos_encoder.transform(df_encoded['Phosphorescence'])
    df_encoded['Label'] = label_encoder.transform(df_encoded['Label'])

    X = df_encoded[['Raman', 'Thermal', 'Electric', 'UV', 'Phosphorescence']]
    y = df_encoded['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)

except Exception as e:
    messagebox.showerror("Critical Error", f"Problem loading model: {e}")
    root.destroy()
    exit()

def classify_diamond(raman, thermal, electric, uv, phos):
    uv = uv.lower()
    phos = phos.lower()

    if (1315 <= raman <= 1345 and thermal > 1800 and electric < 1e-10 and uv in ['blue', 'none'] and phos in ['short', 'none']):
        return "Natural Diamond"
    elif (1315 <= raman <= 1345 and thermal > 1500 and electric < 1e-10 and uv == 'orange' and phos == 'long'):
        return "Lab-grown Diamond"
    elif (800 <= raman <= 835 and 1000 <= thermal <= 1800 and 1e-10 <= electric <= 1e-6 and uv == 'none' and phos == 'long'):
        return "Likely Moissanite"
    else:
        return "Fake"

def predict():
    try:
        if not all([entry_raman.get(), entry_thermal.get(), entry_electric.get(), uv_var.get(), phos_var.get()]):
            messagebox.showwarning("Input Missing", "Please complete all fields.")
            return

        progress['value'] = 0
        root.update_idletasks()

        for i in range(0, 101, 10):
            progress['value'] = i
            root.update()
            time.sleep(0.03)

        raman = float(entry_raman.get())
        thermal = float(entry_thermal.get())
        electric = float(entry_electric.get())
        uv = uv_var.get().capitalize()
        phos = phos_var.get().capitalize()

        rule_result = classify_diamond(raman, thermal, electric, uv, phos)

        user_encoded = [
            raman,
            thermal,
            electric,
            uv_encoder.transform([uv])[0],
            phos_encoder.transform([phos])[0]
        ]

        ai_prediction = model.predict([user_encoded])
        ai_label = label_encoder.inverse_transform(ai_prediction)[0]

        final_result = "💎 Diamond is Authentic!" if rule_result == ai_label else "⚠️ Mismatch Detected!"

        messagebox.showinfo("DiaSureX Result",
                            f"🧠 Rule-based: {rule_result}\n"
                            f"🤖 AI Model: {ai_label}\n"
                            f"🎯 Accuracy: {round(model_accuracy * 100, 2)}%\n\n"
                            f"{final_result}")

    except Exception as e:
        progress['value'] = 0
        messagebox.showerror("Error", f"Input Error: {e}")

header = tk.Label(root, text="DiaSureX", font=("Orbitron", 28, "bold"), bg="#F1F8E9", fg="#00796B")
header.pack(pady=10)

subtitle = tk.Label(root, text="Ultra-Intelligent Diamond Tester", font=("Orbitron", 14), bg="#F1F8E9", fg="#555555")
subtitle.pack(pady=5)

def create_label(text):
    return tk.Label(root, text=text, font=("Poppins", 12, "bold"), bg="#F1F8E9", fg="#333333")

def create_entry():
    return tk.Entry(root, font=("Poppins", 12), bg="#E0F2F1", fg="#00796B", insertbackground="#00796B", relief="flat", highlightthickness=1, highlightbackground="#00796B")

create_label("Raman Shift (cm⁻¹)").pack(pady=5)
entry_raman = create_entry()
entry_raman.pack()

create_label("Thermal Conductivity (W/m·K)").pack(pady=5)
entry_thermal = create_entry()
entry_thermal.pack()

create_label("Electric Conductivity (S/m)").pack(pady=5)
entry_electric = create_entry()
entry_electric.pack()

create_label("UV Fluorescence").pack(pady=5)
uv_options = ["None", "Blue", "Orange", "Variable"]
uv_var = tk.StringVar(value=uv_options[0])
uv_menu = ttk.Combobox(root, textvariable=uv_var, values=uv_options, font=("Poppins", 12))
uv_menu.pack()

create_label("Phosphorescence").pack(pady=5)
phos_options = ["None", "Short", "Long", "Variable"]
phos_var = tk.StringVar(value=phos_options[0])
phos_menu = ttk.Combobox(root, textvariable=phos_var, values=phos_options, font=("Poppins", 12))
phos_menu.pack()

predict_btn = tk.Button(root, text="🔎 Predict Diamond Type", font=("Orbitron", 14, "bold"), bg="#00796B", fg="white", activebackground="#004D40", activeforeground="white", relief="flat", command=predict)
predict_btn.pack(pady=20)

progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate", style="TProgressbar")
progress.pack(pady=15)

style = ttk.Style(root)
style.configure("TProgressbar", troughcolor="#C8E6C9", background="#00796B", thickness=20)

root.mainloop()
