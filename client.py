from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import joblib
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

MONGODB_URI = "mongodb://localhost:27017"
DB_NAME = "smart_city"

state_predictor = None
anomaly_model = None
fault_model = None
mongo = None
db = None

def load_models():
    global state_predictor, anomaly_model, fault_model, mongo, db
    if state_predictor is None:
        print("Loading ML models...")
        state_predictor = joblib.load('state_predictor.pkl')
        anomaly_model = joblib.load('anomaly_model.pkl')
        fault_model = joblib.load('fault_model.pkl')
        print("✓ Models loaded")
    if mongo is None:
        mongo = MongoClient(MONGODB_URI)
        db = mongo[DB_NAME]
        print("✓ Connected to MongoDB")

@app.route('/assets', methods=['GET'])
def get_assets():
    try:
        load_models()
        assets = list(db['ecl'].find(
            {},
            {'_id': 0, 'asset_id': 1, 'street': 1, 'zone': 1, 'wattage': 1}
        ).sort('asset_id', 1))
        return jsonify({
            'success': True,
            'count': len(assets),
            'assets': assets
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_models()
        data = request.json

        asset_id = data.get('asset_id')
        power_w = float(data.get('power_w', 0))

        # Weather features (NO visibility)
        weather = data.get('weather', 'clear')
        precipitation = float(data.get('precipitation', 0))
        temperature = float(data.get('temperature', 20))

        asset = db['ecl'].find_one({'asset_id': asset_id})
        if not asset:
            return jsonify({
                'success': False,
                'error': f'Asset ID "{asset_id}" not found'
            }), 400

        street = asset.get('street', 'Unknown')
        zone = asset.get('zone', 'unknown')
        wattage = asset.get('wattage', 100.0)

        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        is_night = int((hour >= 18) or (hour < 6))

        weather_map = {'clear': 0, 'cloudy': 1, 'rain': 2, 'fog': 3}
        weather_encoded = weather_map.get(weather.lower(), 0)

        heavy_precipitation = int(precipitation > 5.0)
        is_fog = int(weather.lower() == 'fog')
        weather_severity = heavy_precipitation * 0.5 + is_fog * 0.5

        historical = list(db['user_data'].find(
            {'asset_id': asset_id}
        ).sort('ts', -1).limit(10))

        if historical and len(historical) >= 3:
            power_rolling_mean = float(np.mean([h.get('power_w', 0) for h in historical]))
            power_rolling_std = float(np.std([h.get('power_w', 0) for h in historical]))
        else:
            power_rolling_mean = float(power_w)
            power_rolling_std = 5.0

        # State prediction (NO dim)
        X_state = np.array([[
            hour,
            day_of_week,
            power_rolling_mean,
            power_rolling_std,
            precipitation,
            weather_encoded,
            temperature
        ]])

        predicted_state_binary = int(state_predictor.predict(X_state)[0])
        state_proba = state_predictor.predict_proba(X_state)[0]
        state_confidence = float(max(state_proba))
        predicted_state_label = 'ON' if predicted_state_binary == 1 else 'OFF'

        inferred_state = 1 if power_w > 50 else 0
        inferred_state_label = 'ON' if inferred_state == 1 else 'OFF'

        state_mismatch = (inferred_state != predicted_state_binary)

        # Anomaly detection (NO dim)
        expected_power_base = float(is_night * 100)
        weather_boost = weather_severity * 20
        expected_power = expected_power_base + weather_boost
        power_deviation = float(abs(power_w - expected_power))

        irregular_on = int((predicted_state_binary == 1) and (is_night == 0) and (heavy_precipitation == 0) and (is_fog == 0))
        irregular_off = int((predicted_state_binary == 0) and (is_night == 1))

        X_anomaly = np.array([[
            power_w,
            power_rolling_mean,
            power_rolling_std,
            power_deviation,
            irregular_on,
            irregular_off,
            hour,
            precipitation,
            weather_encoded
        ]])

        anomaly_pred = int(anomaly_model.predict(X_anomaly)[0])
        anomaly_label = 'anomaly' if anomaly_pred == -1 else 'normal'
        anomaly_score = float(anomaly_model.decision_function(X_anomaly)[0])

        # Fault prediction (NO dim)
        X_fault = np.array([[
            power_w,
            hour,
            day_of_week,
            is_night,
            power_rolling_mean,
            power_rolling_std,
            power_deviation,
            irregular_on,
            irregular_off,
            int(state_mismatch),
            precipitation,
            weather_encoded,
            temperature
        ]])

        fault_proba = fault_model.predict_proba(X_fault)[0]
        fault_probability = float(fault_proba[1])

        severity = 'normal'
        if anomaly_label == 'anomaly' and fault_probability > 0.7:
            severity = 'critical'
        elif anomaly_label == 'anomaly' or fault_probability > 0.5:
            severity = 'warning'

        # Weather impact
        weather_impact = ""
        if heavy_precipitation:
            weather_impact = "Heavy precipitation - enhanced lighting for safety"
        elif is_fog:
            weather_impact = "Foggy conditions - maximum brightness recommended"
        else:
            weather_impact = "Normal weather conditions"

        record = {
            'asset_id': asset_id,
            'street': street,
            'zone': zone,
            'wattage': wattage,
            'power_w': power_w,
            'inferred_state': inferred_state,
            'predicted_state': predicted_state_label,
            'ts': now,
            'state_confidence': round(state_confidence, 4),
            'state_mismatch': state_mismatch,
            'anomaly_label': anomaly_label,
            'anomaly_score': round(anomaly_score, 4),
            'fault_probability': round(fault_probability, 4),
            'severity': severity,
            'hour': hour,
            'is_night': bool(is_night),
            'weather': weather,
            'precipitation': precipitation,
            'temperature': temperature,
            'weather_severity': round(weather_severity, 2)
        }

        db['user_data'].insert_one(record)

        result = {
            'success': True,
            'message': 'Data analyzed successfully',
            'asset_info': {
                'asset_id': asset_id,
                'street': street,
                'zone': zone,
                'wattage': wattage
            },
            'prediction': {
                'predicted_state': predicted_state_label,
                'state_confidence': round(state_confidence, 3),
                'inferred_state': inferred_state_label,
                'state_match': not state_mismatch,
                'anomaly_label': anomaly_label,
                'anomaly_score': round(anomaly_score, 4),
                'fault_probability': round(fault_probability, 4),
                'severity': severity
            },
            'analysis': {
                'is_night': bool(is_night),
                'expected_power': round(expected_power, 2),
                'power_deviation': round(power_deviation, 2),
                'irregular_behavior': bool(irregular_on or irregular_off),
                'submitted_power': power_w
            },
            'weather': {
                'condition': weather,
                'precipitation': precipitation,
                'temperature': temperature,
                'severity': round(weather_severity, 2),
                'impact': weather_impact
            }
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'message': 'ML Prediction Service Running',
        'features': 'Weather-aware (NO dimming, NO visibility)'
    })

if __name__ == '__main__':
    load_models()
    print("="*60)
    print("🤖 Smart Streetlight ML Prediction Service")
    print("="*60)
    print("✓ Weather-aware predictions (NO dimming)")
    print("✓ Service running on http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
