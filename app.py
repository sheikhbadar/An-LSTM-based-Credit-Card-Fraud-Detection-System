from flask import Flask, render_template, request, jsonify
from datetime import datetime
from model import FraudDetectionModel

app = Flask(__name__)

# Initialize the model
model = FraudDetectionModel()

# Store historical transactions (in a real application, this would be in a database)
historical_transactions = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Get user ID from the request (in a real application, this would come from authentication)
        user_id = data.get('user_id', 'default_user')
        
        # Get historical transactions for this user
        user_history = historical_transactions.get(user_id, [])
        
        # Get prediction and risk factors from our model
        prediction = model.predict(data, user_history)
        risk_factors = model.get_risk_factors(data)
        
        # Update historical transactions
        user_history.append(data)
        if len(user_history) > 5:  # Keep only the last 5 transactions
            user_history = user_history[-5:]
        historical_transactions[user_id] = user_history
        
        # Use a lower threshold for fraud detection (0.4 instead of 0.5)
        return jsonify({
            'prediction': 'Fraudulent' if prediction > 0.4 else 'Not Fraudulent',
            'confidence': float(prediction),
            'risk_factors': risk_factors
        })
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")  # Add logging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
