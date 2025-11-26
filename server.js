const express = require('express');
const { MongoClient } = require('mongodb');
const path = require('path');
const fetch = require('node-fetch');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

const MONGODB_URI = process.env.MONGODB_URI || "mongodb://localhost:27017";
const DB_NAME = "smart_city";

let db;

async function connectDB() {
    try {
        const client = await MongoClient.connect(MONGODB_URI);
        db = client.db(DB_NAME);
        console.log('✓ Connected to MongoDB');
    } catch (error) {
        console.error('MongoDB connection error:', error);
        process.exit(1);
    }
}

app.use(express.static('public'));
app.use(express.json());
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

app.get('/', (req, res) => {
    res.render('dashboard');
});

// API: Submit User Streetlight Data with Weather (NO dim)
app.post('/api/submit_data', async (req, res) => {
    try {
        const data = req.body;
        
        // Check only for required fields: asset_id and power_w (no dim)
        if (!data.asset_id || data.power_w === undefined) {
            return res.status(400).json({ 
                error: 'Missing required fields: asset_id, power_w' 
            });
        }
        
        // Construct the data to send to prediction service (NO dim)
        const submissionData = {
            asset_id: data.asset_id,
            power_w: data.power_w,
            weather: data.weather || 'clear',
            precipitation: data.precipitation !== undefined ? data.precipitation : 0,
            temperature: data.temperature !== undefined ? data.temperature : 20
        };
        
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(submissionData)
        });
        
        const result = await response.json();
        res.json(result);
        
    } catch (error) {
        console.error('Error submitting data:', error);
        res.status(500).json({ 
            error: 'Failed to submit data',
            details: error.message 
        });
    }
});

app.get('/api/user_data', async (req, res) => {
    try {
        const userData = await db.collection('user_data')
            .find({})
            .sort({ ts: -1 })
            .limit(100)
            .toArray();
        
        res.json(userData);
    } catch (error) {
        console.error('Error fetching user data:', error);
        res.status(500).json({ error: 'Failed to fetch user data' });
    }
});

app.get('/api/stats', async (req, res) => {
    try {
        const total = await db.collection('user_data').countDocuments();
        const anomalies = await db.collection('user_data').countDocuments({ anomaly_label: 'anomaly' });
        const highRisk = await db.collection('user_data').countDocuments({ fault_probability: { $gt: 0.5 } });
        
        const lightsOn = await db.collection('user_data').countDocuments({ inferred_state: 1 });
        const lightsOff = total - lightsOn;
        
        res.json({
            total_submissions: total,
            anomalies: anomalies,
            high_risk_faults: highRisk,
            lights_on: lightsOn,
            lights_off: lightsOff
        });
    } catch (error) {
        console.error('Error fetching stats:', error);
        res.status(500).json({ error: 'Failed to fetch statistics' });
    }
});

connectDB().then(() => {
    app.listen(PORT, '0.0.0.0', () => {
        console.log(`\n${'='.repeat(60)}`);
        console.log('🔦 Smart Streetlight Dashboard');
        console.log(`${'='.repeat(60)}`);
        console.log(`✓ Server running on http://localhost:${PORT}`);
        console.log(`✓ Connected to MongoDB: ${DB_NAME}`);
        console.log(`✓ Weather features: precipitation, temperature`);
        console.log(`\nMake sure prediction service is running:`);
        console.log(`  python prediction_service.py`);
        console.log(`${'='.repeat(60)}\n`);
    });
});
