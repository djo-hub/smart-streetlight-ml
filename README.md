# Smart Streetlight ML System 🌃🔦

A weather-aware, energy-efficient smart streetlight analysis and monitoring platform using Python, Flask, Node.js, Mosquitto and MongoDB.

---

## 🚀 Features

- Predict ON/OFF status for streetlights using ML powered by weather, time, and power data.
- Detects anomalies and predicts high-risk faults in real streetlight sensor data.
- Interactive dashboard for manual data submission, result visualization, and history tracking.
- Modern Node.js + EJS web frontend.
- Modular backend: Flask for ML/AI, MongoDB for storage.

---

## 🏗 Project Structure
.
├── dashboard.ejs # Dashboard frontend (HTML/JS, EJS)

├── server.js # Node.js API/dashboard backend

├── client.py # ML prediction Flask API

├── train.py # ML model training pipeline

├── server.py # Data seeder/simulator

├── requirements.txt # Python dependencies

├── package.json # Node.js dependencies

├── .gitignore # Git ignore rules

├── HOW_TO_RUN.txt # Full setup and usage instructions

└── README.md # Project overview (this file)
---
## 🧠 Machine Learning Overview 

Training Pipeline :

Data Generation & Collection :

Historical on/off states and power readings are generated for 145 real streetlights covering 72 hours, along with realistic weather conditions, using a custom seeder.
Data is stored in MongoDB (status, ecl collections).

Feature Engineering:

The following statistical and context-based features are created for each record:
Power consumption (power_w)
Time of day (hour, day_of_week)
Is Night/Day (is_night)
Rolling mean & std of power (power_rolling_mean, power_rolling_std)
Expected power based on weather and time
Power deviation from expected
Anomalous patterns (irregular_on, irregular_off)
Weather: weather (encoded), precipitation, temperature, weather_severity

Anomaly Detection :

Algorithm: IsolationForest (unsupervised)
Features: power_w, power_rolling_mean, power_rolling_std,power_deviation, irregular_on, irregular_off, hour,precipitation, weather_encoded
Finds streetlight readings that differ strongly from normal past behavior.

Fault Prediction :

Algorithm: RandomForestClassifier (supervised)
Features: power_w, hour, day_of_week, is_night,power_rolling_mean, power_rolling_std, power_deviation,irregular_on, irregular_off, state_changed,precipitation, weather_encoded, temperature
Predicts the probability that a reading indicates a serious or upcoming fault.
Evaluated using accuracy, precision, recall, and F1-score.

State Prediction :

Algorithm: RandomForestClassifier (supervised)
Features: hour, day_of_week, power_rolling_mean, power_rolling_std,precipitation, weather_encoded, temperature
Enables validation of observed readings and model-based alerts.

Algorithms Used :
Isolation Forest: Outlier detection for spotting sensor or behavioral anomalies.​
Random Forest Classifier: Fault prediction and ON/OFF state prediction for robust interpretability.​

## ⚡️ Getting Started

**1. Clone this repository:**
git clone https://github.com/djo-hub/smart-streetlight-ml.git
cd smart-streetlight-ml


**2. Install dependencies:**
pip install -r requirements.txt
npm install


**3. MongoDB**
- Make sure MongoDB is installed and running on `localhost:27017`.

**4. Seed the database:**
python  server.py

**5. Train models:**
python train.py

**6. Start prediction API (Terminal 1):**
python client.py

**7. Start dashboard server (Terminal 2):**
node server.js

**8. Open in browser:**
http://localhost:3000


---

## 📋 How to Use

1. Select a streetlight asset from dropdown.
2. Enter a real or simulated power value and weather conditions.
3. Submit to receive predictions (ON/OFF, Anomaly, Fault Risk).
4. View your analysis in the history table below the form.

---

## 📦 Requirements

- Python 3.8+
- Node.js 16+
- MongoDB
- See `requirements.txt` and `package.json` for full details.

---

## ⭐ Acknowledgments

- [OpenWeather](https://openweathermap.org/) for weather API when used.
- Washington DC open civic streetlight data.

---

*Built by Djoghlal Abid , Seghiri Med Islem , Hernane DhiaaEddine.*

