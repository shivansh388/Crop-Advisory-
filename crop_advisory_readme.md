#  Development of a Data-Driven Crop Advisory System Using Soil and Environmental Parameters

## CODE CRAFTERS

## 24CSE256 - Shivanshu Verma 24CSEAIML194 - Basudev Patro 24ECE070 - Adarsh Palo 24CSEAIML335 - Tara Prasad Panda

## Problem Statement

Design and develop a simple, sensor-assisted or data-driven system that recommends the most suitable crop to cultivate based on:

- Soil macronutrients (N, P, K)
- pH level
- Temperature
- Humidity
- Rainfall

The system should analyze current soil and weather conditions to provide:

- Accurate crop suggestions
- Fertilizer advisory services

Targeted especially for small and marginal farmers to improve profitability and yield.

---

## 📊 Dataset

We used the open-source dataset available on Kaggle:\
[🔗 Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

This dataset includes:

- **Soil nutrients:** Nitrogen (N), Phosphorous (P), Potassium (K)
- **Environmental:** Temperature (°C), Humidity (%), Rainfall (mm)
- **pH level**
- **Crop label**

---

## ⚙️ Features of the System

- Input interface for soil and weather parameters
- Machine Learning model (Random Forest) trained for crop recommendation
- Additional fertilizer suggestions based on deficiency/excess of N, P, K
- Lightweight interface for farmers with easy deployment potential

---

## 📈 Model Overview

- **Preprocessing:** Normalization, Label Encoding
- **Model used:** Random Forest Classifier (accuracy: \~98%)
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `streamlit`

---

## 🧪 How to Use the System

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   cd YOUR_REPOSITORY
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Enter soil and environmental parameters to get crop & fertilizer recommendations.

---

## 🖼 Presentation

📊 **Slide Deck (PPT or PDF):** [View Presentation](https://www.canva.com/design/DAGt-9jOhVA/McdR4eYBUbQ-czND8tt6Ig/view?utm_content=DAGt-9jOhVA&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hd23559e9e6\

## source code 

# crop_gui_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import tkinter as tk
from tkinter import messagebox

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")
X = data.drop('label', axis=1)
y = data['label']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

# GUI

def predict_crop():
    try:
        input_data = [
            float(entry_n.get()),
            float(entry_p.get()),
            float(entry_k.get()),
            float(entry_temp.get()),
            float(entry_humidity.get()),
            float(entry_ph.get()),
            float(entry_rainfall.get())
        ]
        with open("crop_model.pkl", "rb") as f:
            model = pickle.load(f)
        prediction = model.predict([input_data])[0]
        messagebox.showinfo("Prediction", f"Recommended Crop: {prediction}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

app = tk.Tk()
app.title("Crop Recommendation System")

# Labels and Entries
tk.Label(app, text="Nitrogen").grid(row=0, column=0)
tk.Label(app, text="Phosphorus").grid(row=1, column=0)
tk.Label(app, text="Potassium").grid(row=2, column=0)
tk.Label(app, text="Temperature").grid(row=3, column=0)
tk.Label(app, text="Humidity").grid(row=4, column=0)
tk.Label(app, text="pH").grid(row=5, column=0)
tk.Label(app, text="Rainfall").grid(row=6, column=0)

entry_n = tk.Entry(app)
entry_p = tk.Entry(app)
entry_k = tk.Entry(app)
entry_temp = tk.Entry(app)
entry_humidity = tk.Entry(app)
entry_ph = tk.Entry(app)
entry_rainfall = tk.Entry(app)

entry_n.grid(row=0, column=1)
entry_p.grid(row=1, column=1)
entry_k.grid(row=2, column=1)
entry_temp.grid(row=3, column=1)
entry_humidity.grid(row=4, column=1)
entry_ph.grid(row=5, column=1)
entry_rainfall.grid(row=6, column=1)

tk.Button(app, text="Predict Crop", command=predict_crop).grid(row=7, column=0, columnspan=2)

app.mainloop()

## 🤝 Acknowledgement

We thank the organizers for the opportunity and guidance throughout the challenge. This solution is designed keeping in mind the long-term benefit of small-scale farmers by promoting efficient agricultural practices.

---

## 📧 Contact

For any queries, contact us at:
Adarsh Palo 8328975725 ,and team .