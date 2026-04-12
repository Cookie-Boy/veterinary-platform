from flask import Flask, request, jsonify
from minio import Minio
import joblib
import pandas as pd
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# MinIO configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'ml-models')
MINIO_MODEL_OBJECT = os.getenv('MINIO_MODEL_OBJECT', 'pet_health_model.pkl')

MODEL_PATH = "pet_health_model.pkl"
HTTP_PORT = 8091

# Download model from MinIO on startup
def load_model_from_minio():
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        local_path = '/tmp/model.pkl'
        client.fget_object(MINIO_BUCKET, MINIO_MODEL_OBJECT, local_path)
        model = joblib.load(local_path)
        app.logger.info("Model loaded successfully from MinIO")
        return model
    except Exception as e:
        app.logger.error(f"Failed to load model from MinIO: {e}")
        raise e
    
def load_model_local():
    try:
        model = joblib.load(MODEL_PATH)
        app.logger.info(f"Model loaded successfully from local file: {MODEL_PATH}")
        return model
        
    except FileNotFoundError:
        app.logger.error(f"Model file not found at {MODEL_PATH}")
        raise
    except Exception as e:
        app.logger.error(f"Failed to load model: {e}")
        raise e

# model = load_model_from_minio()
model = load_model_local()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    app.logger.info(f"Received prediction request with data: {data}")
    
    # Expected fields: species, breed, heartRate, respiration, temperature
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0].tolist()
    
    # Формируем объект для ответа
    response_data = {
        'anomalyClass': int(pred), 
        'probabilities': proba
    }
    
    # Логируем результат
    app.logger.info(f"Returning prediction response: {response_data}")
    app.logger.info(f"Predicted class: {int(pred)}, Probabilities: {proba}")
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=True, use_reloader=False)