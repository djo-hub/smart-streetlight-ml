# Smart Streetlight ML System 🌃🔦

A weather-aware, energy-efficient smart streetlight analysis and monitoring platform using Python, Flask, Node.js, and MongoDB.

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

## 🛡 Security & Contributions

- No credentials or sensitive keys should be checked into this repo. See `.gitignore`.

## ⭐ Acknowledgments

- [OpenWeather](https://openweathermap.org/) for weather API when used.
- Washington DC open civic streetlight data.

---

*Built by Djoghlal Abid , Seghiri Med Islam , Harnane DhiaaEddine.*

