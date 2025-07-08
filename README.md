# NYC Cab Demand Prediction

This project aims to predict the **cab demand in different zones of New York City** using historical trip data, weather information, and various machine learning models. It helps to optimize cab availability and improve service efficiency.

---

## Project Structure
Cab-Demand-Prediction/
├── Code/
│ ├── Backend/
│ │ ├── API.py
│ │ ├── train_ml.py
│ │ └── fetch_historical_weather.py
│ ├── Frontend/
│ │ ├── index.html
│ │ ├── style.css
│ │ ├── script.js
│ │ └── accuary.py
├── .gitignore
├── README.md

---

## 💡 Features

- 🔍 Predicts cab demand per zone using historical FHV data
- 🌦 Integrates weather data to improve prediction accuracy
- 📊 Evaluates performance with MAE, RMSE, and zone-wise metrics
- 🖥️ Simple frontend visualization with charts and stats
- 🚀 REST API for prediction (Flask-based backend)

---

## 🔗 Dataset Source

We use NYC’s open public dataset for cab rides:

👉 [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

The dataset includes detailed trip-level records like pickup time, location, and more.

---

## 🛠️ Tech Stack

- **Python** – Data preprocessing & ML model training
- **LightGBM** – Machine learning model for demand prediction
- **Pandas / NumPy** – Data analysis
- **Parquet** – Efficient data storage
- **Flask** – Backend API
- **HTML / CSS / JS** – Frontend interface
- **Chart.js** – Graphs and charts

---

## 🧹 Excluded from GitHub

To keep the repository lightweight and within GitHub’s limits:

- 📁 `Code/Dataset/` – Raw trip data files
- 📁 `Code/Backend/models_ml/` – Trained ML models
- 📁 `Code/Backend/processed_data_ml/` – Processed intermediate data
- 📁 `Documentation/` – PDF/DOCX reports and presentations
- 📄 Large files over 100 MB (handled with `.gitignore`)

---
