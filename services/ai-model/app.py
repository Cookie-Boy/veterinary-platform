from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = "pet_health_model.pkl"
HTTP_PORT = 8091
    
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

model = load_model_local()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    app.logger.info(f"Received prediction request with data: {data}")
    
    # Expected fields: species, breed, heartRate, respiration, temperature
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0].tolist()
    
    response_data = {
        'anomalyClass': int(pred), 
        'probabilities': proba
    }
    
    app.logger.info(f"Returning prediction response: {response_data}")
    app.logger.info(f"Predicted class: {int(pred)}, Probabilities: {proba}")
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=True, use_reloader=False)