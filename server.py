import requests
from datetime import datetime, timezone, timedelta
import random
from pymongo import MongoClient, GEOSPHERE, UpdateOne, ASCENDING, DESCENDING

# ArcGIS layer and fields (DC Streetlights)
ARCGIS_LAYER = "https://maps2.dcgis.dc.gov/dcgis/rest/services/DDOT/Streetlights/FeatureServer/0"
FIELDS = ["FACILITYID","WATTAGE1","LIGHTTYPE","STREETNAME","WARD"]
LIMIT = 2000
WHERE_FILTER = None
ENVELOPE = None

# History configuration
HOURS = 72
STEP_MINUTES = 60

# MongoDB
MONGODB_URI = "mongodb://localhost:27017"
DB_NAME = "smart_city"
COLL_ASSETS = "ecl"
COLL_STATUS = "status"

def to_float(x, default=None):
    try:
        if x is None: return default
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip().split()[0].replace(",", ".")
        return float(s)
    except Exception:
        return default

def arcgis_query_2k():
    params = {
        "where": WHERE_FILTER or "1=1",
        "outFields": ",".join(FIELDS),
        "outSR": "4326",
        "resultOffset": 0,
        "resultRecordCount": LIMIT,
        "f": "geojson"
    }
    if ENVELOPE:
        xmin, ymin, xmax, ymax = ENVELOPE
        params.update({
            "geometry": f"{xmin},{ymin},{xmax},{ymax}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects"
        })
    r = requests.get(f"{ARCGIS_LAYER}/query", params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def normalize_feature(feature):
    props = feature.get("properties", {}) or {}
    geom = feature.get("geometry", {}) or {}
    asset_id = props.get("FACILITYID")
    if asset_id is None:
        return None
    ward = props.get("WARD")
    street = props.get("STREETNAME")
    watt = to_float(props.get("WATTAGE1"), default=100.0)
    ltype = props.get("LIGHTTYPE")
    location = None
    if geom.get("type") == "Point" and isinstance(geom.get("coordinates"), list):
        location = {"type": "Point", "coordinates": geom["coordinates"]}
    return {
        "asset_id": str(asset_id),
        "provider": "DC_DDOT",
        "zone": str(ward) if ward is not None else "unknown",
        "street": street,
        "light_type": ltype,
        "wattage": watt,
        "location": location,
        "raw": props
    }

def ensure_indexes(db):
    db[COLL_ASSETS].create_index("asset_id", unique=True)
    db[COLL_ASSETS].create_index([("location", GEOSPHERE)])
    db[COLL_STATUS].create_index([("asset_id", ASCENDING), ("ts", DESCENDING)], background=True)
    db[COLL_STATUS].create_index([("street", ASCENDING), ("ts", DESCENDING)], background=True)

def generate_weather(hour, day_offset=0):
    """
    Generate realistic weather data WITHOUT visibility.
    """
    random.seed(day_offset * 1000 + hour)
    if 6 <= hour <= 18:
        weather_options = ['clear', 'cloudy', 'rain', 'fog']
        weights = [0.60, 0.25, 0.10, 0.05]
    else:
        weather_options = ['clear', 'cloudy', 'rain', 'fog']
        weights = [0.50, 0.20, 0.15, 0.15]
    weather = random.choices(weather_options, weights=weights)[0]
    # Precipitation (mm/hour)
    if weather in ['clear', 'cloudy']:
        precipitation = 0.0
    elif weather == 'rain':
        precipitation = random.uniform(0.5, 15.0)
    else:  # fog (light drizzle)
        precipitation = random.uniform(0.0, 0.5)
    # Temperature (°C)
    if 6 <= hour <= 18:
        temperature = random.uniform(10, 30)
    else:
        temperature = random.uniform(0, 20)
    random.seed()
    return {
        'weather': weather,
        'precipitation': round(precipitation, 2),
        'temperature': round(temperature, 1)
    }

def generate_history_for_asset(doc, start_ts, points, step_minutes):
    """
    Generate 72-hour history with weather features (NO dim field).
    """
    hist = []
    aid = doc["asset_id"]
    ward = doc.get("zone") or "unknown"
    street = doc.get("street")
    watt = to_float(doc.get("wattage"), default=100.0) or 100.0
    ts = start_ts
    for i in range(points):
        hour = ts.hour
        day_offset = i // 24
        weather_data = generate_weather(hour, day_offset)
        base_on = (hour >= 18 or hour < 6)
        heavy_rain = weather_data['precipitation'] > 5.0
        is_fog = weather_data['weather'] == 'fog'
        weather_boost = False
        if (15 <= hour <= 17 or 6 <= hour <= 8) and (heavy_rain or is_fog):
            weather_boost = random.random() < 0.3
        flip = random.random() < 0.04
        state = "on" if ((base_on or weather_boost) ^ flip) else "off"
        if state == "on":
            power = watt * (0.9 + 0.2*random.random())
        else:
            power = 0.2*random.random()
        hist.append({
            "asset_id": aid,
            "zone": ward,
            "street": street,
            "ts": ts,
            "state": state,
            "inferred_state": None,
            "confidence": None,
            "power_w": round(float(power), 2),
            # Weather features
            "weather": weather_data['weather'],
            "precipitation": weather_data['precipitation'],
            "temperature": weather_data['temperature'],
            "source": "seed-72h-weather"
        })
        ts = ts + timedelta(minutes=step_minutes)
    return hist

def deduplicate_by_street(docs):
    seen = set()
    unique = []
    for d in docs:
        street = (d.get("street") or "").strip() if d.get("street") else ""
        if not street:
            continue
        key = street.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique

def main():
    mongo = MongoClient(MONGODB_URI)
    db = mongo[DB_NAME]
    ensure_indexes(db)
    fc = arcgis_query_2k()
    feats = fc.get("features", [])
    print(f"Fetched {len(feats)} features")
    normalized_all = [normalize_feature(f) for f in feats if normalize_feature(f)]
    unique_by_street = deduplicate_by_street(normalized_all)
    print(f"After unique-by-street filter: {len(unique_by_street)} assets (one per street)")
    assets_coll = db[COLL_ASSETS]
    ops = []
    for doc in unique_by_street:
        ops.append(UpdateOne({"asset_id": doc["asset_id"]}, {"$set": doc}, upsert=True))
    if ops:
        res = assets_coll.bulk_write(ops, ordered=False)
        upserts = (res.upserted_count or 0) + (res.modified_count or 0)
        print(f"Upserted/updated ~{upserts} assets into {COLL_ASSETS}")
    points = int(HOURS * (60 // STEP_MINUTES))
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(minutes=STEP_MINUTES * (points - 1))
    print(f"Generating {points} points per asset (with weather) from {start.isoformat()} to {end.isoformat()}")
    status_coll = db[COLL_STATUS]
    batch = []
    BATCH_SIZE = 5000
    total = 0
    for doc in unique_by_street:
        batch.extend(generate_history_for_asset(doc, start, points, STEP_MINUTES))
        if len(batch) >= BATCH_SIZE:
            status_coll.insert_many(batch, ordered=False)
            total += len(batch)
            print(f"  Inserted {total} status rows so far...")
            batch = []
    if batch:
        status_coll.insert_many(batch, ordered=False)
        total += len(batch)
    print(f"✓ Inserted {total} status rows into {COLL_STATUS}")
    print(f"✓ Weather features included: weather, precipitation, temperature")
    sample = status_coll.find_one()
    if sample:
        print(f"\nSample record with weather:")
        print(f"  Time: {sample['ts']}")
        print(f"  Weather: {sample['weather']}")
        print(f"  Precipitation: {sample['precipitation']}mm/h")
        print(f"  Temperature: {sample['temperature']}°C")
        print(f"  State: {sample['state']}, Power: {sample['power_w']}W")

if __name__ == "__main__":
    main()
