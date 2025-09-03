from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

ML_SERVICE_URL = "http://ml-service:5001"   # container-to-container hostname

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models')
def get_models():
    try:
        resp = requests.get(f"{ML_SERVICE_URL}/models")
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        resp = requests.post(f"{ML_SERVICE_URL}/predict", json=data)
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info')
def model_info():
    # This can hit another ML-service endpoint for accuracy, confusion matrix, etc.
    # For now, placeholder:
    return jsonify({'info': 'Model info placeholder'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)