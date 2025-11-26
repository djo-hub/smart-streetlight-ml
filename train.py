import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timezone, timedelta
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# MongoDB Configuration
MONGODB_URI = "mongodb://localhost:27017"
DB_NAME = "smart_city"
COLL_STATUS = "status"
COLL_ASSETS = "ecl"

class StreetlightMLPipeline:
    def __init__(self):
        self.mongo = MongoClient(MONGODB_URI)
        self.db = self.mongo[DB_NAME]
        self.anomaly_model = None
        self.fault_model = None
        self.state_predictor = None
        self.scaler = StandardScaler()
        
    def fetch_data(self):
        print("Fetching data from MongoDB...")
        cursor = self.db[COLL_STATUS].find({})
        data = list(cursor)
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} records")
        return df
    
    def engineer_features(self, df):
        print("Engineering features ...")

        df['ts'] = pd.to_datetime(df['ts'])
        df['hour'] = df['ts'].dt.hour
        df['day_of_week'] = df['ts'].dt.dayofweek
        df['is_night'] = ((df['hour'] >= 18) | (df['hour'] < 6)).astype(int)
        
        weather_map = {'clear': 0, 'cloudy': 1, 'rain': 2, 'fog': 3}
        df['weather_encoded'] = df['weather'].map(weather_map).fillna(0)
        
        df['weather_severity'] = (
            (df['precipitation'] > 5.0).astype(int) * 0.5 +
            (df['weather'] == 'fog').astype(int) * 0.5
        )
        
        df_sorted = df.sort_values(['asset_id', 'ts'])
        df['power_rolling_mean'] = df.groupby('asset_id')['power_w'].transform(
            lambda x: x.rolling(window=6, min_periods=1).mean()
        )
        df['power_rolling_std'] = df.groupby('asset_id')['power_w'].transform(
            lambda x: x.rolling(window=6, min_periods=1).std()
        )
        
        df['state_binary'] = (df['state'] == 'on').astype(int)
        df['prev_state'] = df.groupby('asset_id')['state_binary'].shift(1)
        df['state_changed'] = (df['state_binary'] != df['prev_state']).astype(int)
        
        df['expected_power_base'] = df['is_night'] * 100
        df['weather_boost'] = df['weather_severity'] * 20
        df['expected_power'] = df['expected_power_base'] + df['weather_boost']
        df['power_deviation'] = np.abs(df['power_w'] - df['expected_power'])
        
        df['heavy_precipitation'] = (df['precipitation'] > 5.0).astype(int)
        df['irregular_on'] = (
            (df['state'] == 'on') & 
            (df['is_night'] == 0) & 
            (df['heavy_precipitation'] == 0) &
            (df['weather'] != 'fog')
        ).astype(int)
        df['irregular_off'] = (
            (df['state'] == 'off') & 
            (df['is_night'] == 1)
        ).astype(int)
        
        df = df.fillna(0)
        print(f"✓ Features engineered. Weather features: weather, precipitation, temperature")
        return df
    
    def detect_anomalies(self, df):
        print("\n=== ANOMALY DETECTION  ===")
        features = [
            'power_w', 'power_rolling_mean', 'power_rolling_std',
            'power_deviation', 'irregular_on', 'irregular_off', 'hour',
            'precipitation', 'weather_encoded'
        ]
        X = df[features].values
        self.anomaly_model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        )
        print("Training Isolation Forest...")
        predictions = self.anomaly_model.fit_predict(X)
        df['anomaly'] = predictions
        df['anomaly_label'] = df['anomaly'].map({1: 'normal', -1: 'anomaly'})
        df['anomaly_score'] = self.anomaly_model.decision_function(X)
        n_anomalies = (predictions == -1).sum()
        print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.2f}%)")
        joblib.dump(self.anomaly_model, 'anomaly_model.pkl')
        print("Saved anomaly model to anomaly_model.pkl")
        return df
    
    def predictive_maintenance(self, df):
        print("\n=== PREDICTIVE MAINTENANCE ===")
        df['fault'] = 0
        fault_conditions = (
            (df['irregular_off'] == 1) |
            (df['power_w'] > 200) |
            (df['power_w'] < 0) |
            (df['state_changed'] == 1) |
            ((df['weather_severity'] > 0.5) & (df['state'] == 'off') & (df['is_night'] == 1))
        )
        df.loc[fault_conditions, 'fault'] = 1

        feature_cols = [
            'power_w', 'hour', 'day_of_week', 'is_night',
            'power_rolling_mean', 'power_rolling_std', 'power_deviation',
            'irregular_on', 'irregular_off', 'state_changed',
            'precipitation', 'weather_encoded', 'temperature'
        ]
        X = df[feature_cols].values
        y = df['fault'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Training Random Forest for fault prediction...")
        self.fault_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.fault_model.fit(X_train, y_train)
        y_pred = self.fault_model.predict(X_test)
        print("\nFault Prediction Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fault']))
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.fault_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        joblib.dump(self.fault_model, 'fault_model.pkl')
        joblib.dump(feature_cols, 'feature_cols.pkl')
        print("Saved fault model to fault_model.pkl")
        df['fault_probability'] = self.fault_model.predict_proba(X)[:, 1]
        return df
    
    def predict_state(self, df):
        print("\n=== STATE PREDICTION ===")
        feature_cols = [
            'hour', 'day_of_week', 'power_rolling_mean', 
            'power_rolling_std',
            'precipitation', 'weather_encoded', 'temperature'
        ]
        X = df[feature_cols].values
        y = df['state_binary'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Training weather-aware state predictor...")
        self.state_predictor = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.state_predictor.fit(X_train, y_train)
        y_pred = self.state_predictor.predict(X_test)
        print("\nState Prediction Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['OFF', 'ON']))
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.state_predictor.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop Features for State Prediction:")
        print(feature_importance.head())
        joblib.dump(self.state_predictor, 'state_predictor.pkl')
        print("Saved state predictor to state_predictor.pkl")
        df['predicted_state'] = self.state_predictor.predict(X)
        df['state_confidence'] = self.state_predictor.predict_proba(X).max(axis=1)
        return df
    
    def save_results(self, df):
        print("\n=== SAVING RESULTS ===")
        results_coll = self.db['ml_results']
        records = df[[
            'asset_id', 'zone', 'street', 'ts', 'state', 'power_w',
            'anomaly_label', 'anomaly_score', 'fault_probability',
            'predicted_state', 'state_confidence',
            'weather', 'precipitation', 'temperature'
        ]].to_dict('records')
        results_coll.delete_many({})
        results_coll.insert_many(records)
        print(f"Saved {len(records)} analysis results to 'ml_results' collection")
    
    def run_pipeline(self):
        print("="*60)
        print("Starting Streetlight ML Pipeline with Weather")
        print("="*60)
        df = self.fetch_data()
        df = self.engineer_features(df)
        df = self.detect_anomalies(df)
        df = self.predictive_maintenance(df)
        df = self.predict_state(df)
        self.save_results(df)
        print("\n" + "="*60)
        print("✓ Pipeline completed successfully!")
        print("✓ Models trained with weather features")
        print("="*60)
        return df

if __name__ == "__main__":
    pipeline = StreetlightMLPipeline()
    results_df = pipeline.run_pipeline()
