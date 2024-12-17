from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://192.168.1.13:8081", "exp://192.168.1.13:8081"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load models
try:
    # Q-Chat-10 model
    with open('qchat_model.pkl', 'rb') as file:
        qchat_model, qchat_scaler, qchat_selector = pickle.load(file)
    
    # AQ-10 model
    with open('aq10_model.pkl', 'rb') as file:
        aq10_model, aq10_scaler, aq10_selector = pickle.load(file)
except Exception as e:
    print(f"Error loading models: {str(e)}")

def get_risk_level(probability):
    if probability < 0.3:
        return "low"
    elif probability < 0.6:
        return "medium"
    else:
        return "high"

@app.route('/predict/qchat', methods=['POST'])
def predict_qchat():
    try:
        print('Received Q-Chat-10 prediction request')
        data = request.json
        print('Request data:', data)
        
        # Match exact feature names from Q-Chat-10 training
        qchat_features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
                         'Age_Mons', 'Sex', 'Jaundice', 'Family_mem_with_ASD']
        
        # Extract features in correct order
        input_features = []
        for feature in qchat_features:
            value = data.get(feature)
            if value is None:
                return jsonify({
                    'error': f'Missing required feature: {feature}',
                    'status': 'error'
                }), 400
            input_features.append(value)
        
        # Process input
        input_array = np.array(input_features).reshape(1, -1)
        input_scaled = qchat_scaler.transform(input_array)
        input_selected = qchat_selector.transform(input_scaled)
        prediction_result = int(qchat_model.predict(input_selected)[0])
        
        # Get prediction
        probabilities = qchat_model.predict_proba(input_selected)[0]
        autism_probability = float(probabilities[1])
        
        # Calculate risk level and confidence
        risk_level = get_risk_level(autism_probability)
        confidence_score = float(abs(autism_probability - 0.5) * 2)
        
        return jsonify({
            'assessment_type': 'Q-Chat-10',
            'autism_probability': autism_probability,
            'result': prediction_result,
            'risk_level': risk_level,
            'confidence_score': confidence_score,
            'status': 'success'
        })
        
    except Exception as e:
        print('Error during Q-Chat-10 prediction:', str(e))
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/predict/aq10', methods=['POST'])
def predict_aq10():
    try:
        print('Received AQ-10 prediction request')
        data = request.json
        print('Request data:', data)
        
        # Match exact feature names from AQ-10 training
        aq10_features = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                        'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                        'age', 'gender', 'jundice', 'austim']
        
        # Extract features in correct order
        input_features = []
        for feature in aq10_features:
            value = data.get(feature)
            if value is None:
                return jsonify({
                    'error': f'Missing required feature: {feature}',
                    'status': 'error'
                }), 400
            input_features.append(value)
        
        # Process input
        input_array = np.array(input_features).reshape(1, -1)
        input_scaled = aq10_scaler.transform(input_array)
        input_selected = aq10_selector.transform(input_scaled)
        prediction_result = int(aq10_model.predict(input_selected)[0])
        # Get prediction
        probabilities = aq10_model.predict_proba(input_selected)[0]
        autism_probability = float(probabilities[1])
        
        # Calculate risk level and confidence
        risk_level = get_risk_level(autism_probability)
        confidence_score = float(abs(autism_probability - 0.5) * 2)
        
        return jsonify({
            'assessment_type': 'AQ-10',
            'autism_probability': autism_probability,
            'result': prediction_result,
            'risk_level': risk_level,
            'confidence_score': confidence_score,
            'status': 'success'
        })
        
    except Exception as e:
        print('Error during AQ-10 prediction:', str(e))
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    return jsonify({
        'qchat_features': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
                          'Age_Mons', 'Sex', 'Jaundice', 'Family_mem_with_ASD'],
        'aq10_features': ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                         'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                         'age', 'gender', 'jundice', 'austim']
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'qchat': hasattr(qchat_model, 'predict'),
            'aq10': hasattr(aq10_model, 'predict')
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)