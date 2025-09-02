from flask import Flask, request, jsonify
import time
from predictor import ImagePredictor

app = Flask(__name__)
predictor = ImagePredictor()  # Models load here once!

@app.route('/health')
def health():
    return {'status': 'healthy', 'models_loaded': len(predictor.models)}

@app.route('/models')  
def get_models():
    return {'available_models': list(predictor.models.keys())}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid or missing JSON body'}), 400

    image = data.get('image')
    model_name = data.get('model_name')
    if image is None or model_name is None:
        return jsonify({'error': 'Both "image" and "model_name" are required'}), 400

    result = predictor.predict(image, model_name)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)