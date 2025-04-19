import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

class FraudLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(FraudLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = h_lstm[:, -1, :]  # Last time step
        out = self.fc(out)
        return self.sigmoid(out)

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.location_encoder = LabelEncoder()
        self.device_encoder = LabelEncoder()
        self.sequence_length = 5  # Number of transactions to consider
        
        # Initialize encoders with common values
        self.location_encoder.fit(['Los Angeles, USA', 'New York, USA', 'Moscow, Russia', 
                                 'Beijing, China', 'Dubai, UAE', 'London, UK', 'Unknown'])
        self.device_encoder.fit(['mobile', 'desktop', 'laptop', 'tablet', 'other'])
    
    def prepare_features(self, transaction_data):
        """Prepare features for a single transaction"""
        features = []
        
        # Amount (normalized)
        amount = float(transaction_data['amount'])
        
        # Time features
        transaction_time = datetime.strptime(transaction_data['time'], '%Y-%m-%dT%H:%M')
        hour = transaction_time.hour
        day_of_week = transaction_time.weekday()
        
        # Encode categorical features with handling for unseen labels
        try:
            location = self.location_encoder.transform([transaction_data['location']])[0]
        except ValueError:
            # If location is not in the encoder, use 'Unknown' category
            location = self.location_encoder.transform(['Unknown'])[0]
        
        try:
            device = self.device_encoder.transform([transaction_data['device']])[0]
        except ValueError:
            # If device is not in the encoder, use 'other' category
            device = self.device_encoder.transform(['other'])[0]
        
        # Create feature vector
        features = np.array([
            amount,
            location,
            device,
            hour,
            day_of_week
        ])
        
        return features
    
    def create_sequence(self, current_transaction, historical_transactions=None):
        """Create a sequence of transactions for LSTM input"""
        if historical_transactions is None:
            historical_transactions = []
        
        # Prepare current transaction features
        current_features = self.prepare_features(current_transaction)
        
        # Prepare historical transaction features
        historical_features = []
        for trans in historical_transactions[-self.sequence_length-1:]:
            historical_features.append(self.prepare_features(trans))
        
        # Combine features
        sequence = historical_features + [current_features]
        
        # Pad or truncate sequence
        if len(sequence) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(sequence), sequence[0].shape[0]))
            sequence = np.vstack([padding, sequence])
        else:
            sequence = sequence[-self.sequence_length:]
        
        return sequence
    
    def train(self, historical_data):
        """Train the LSTM model on historical data"""
        # Prepare sequences and labels
        sequences = []
        labels = []
        
        for user_transactions in historical_data:
            for i in range(len(user_transactions)):
                current_transaction = user_transactions[i]
                historical_transactions = user_transactions[:i]
                
                sequence = self.create_sequence(current_transaction, historical_transactions)
                sequences.append(sequence)
                labels.append(current_transaction.get('is_fraud', 0))
        
        # Convert to tensors
        sequences = torch.FloatTensor(sequences)
        labels = torch.FloatTensor(labels)
        
        # Initialize model
        input_size = sequences.shape[2]  # Number of features
        self.model = FraudLSTM(input_size=input_size)
        
        # Training parameters
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        num_epochs = 50
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # Save the model and encoders
        torch.save(self.model.state_dict(), 'fraud_model.pth')
        joblib.dump(self.scaler, 'scaler.joblib')
        joblib.dump(self.location_encoder, 'location_encoder.joblib')
        joblib.dump(self.device_encoder, 'device_encoder.joblib')
    
    def predict(self, transaction_data, historical_transactions=None):
        """Make prediction for a single transaction"""
        if self.model is None:
            # Load the trained model and encoders
            input_size = 5  # Number of features
            self.model = FraudLSTM(input_size=input_size)
            self.model.load_state_dict(torch.load('fraud_model.pth'))
            self.model.eval()
            
            self.scaler = joblib.load('scaler.joblib')
            self.location_encoder = joblib.load('location_encoder.joblib')
            self.device_encoder = joblib.load('device_encoder.joblib')
        
        # Create sequence
        sequence = self.create_sequence(transaction_data, historical_transactions)
        
        # Convert to tensor and make prediction
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(sequence_tensor).item()
        
        # Adjust prediction based on risk factors
        risk_factors = self.get_risk_factors(transaction_data)
        
        # Calculate risk multiplier based on specific risk factors
        risk_multiplier = 1.0
        for factor in risk_factors:
            if "Extremely high transaction amount" in factor:
                risk_multiplier += 0.4  # Increased from 0.3
            elif "Very high transaction amount" in factor:
                risk_multiplier += 0.3  # Increased from 0.2
            elif "High-risk location detected" in factor:
                risk_multiplier += 0.35  # Increased from 0.25
            elif "Unknown device type used" in factor:
                risk_multiplier += 0.25  # Increased from 0.15
            else:
                risk_multiplier += 0.15  # Increased from 0.1
        
        adjusted_prediction = min(prediction * risk_multiplier, 1.0)
        
        # Lower the threshold for fraud detection
        return adjusted_prediction
    
    def get_risk_factors(self, transaction_data):
        """Get list of risk factors for the transaction"""
        risk_factors = []
        
        # Amount risk (adjusted thresholds)
        amount = float(transaction_data['amount'])
        if amount > 10000:
            risk_factors.append("Extremely high transaction amount (>$10,000)")
        elif amount > 5000:
            risk_factors.append("Very high transaction amount (>$5,000)")
        elif amount > 1000:
            risk_factors.append("High transaction amount (>$1,000)")
        
        # Location risk (more detailed)
        if transaction_data['location'] != transaction_data['usual_location']:
            if transaction_data['location'] in ['Moscow, Russia', 'Beijing, China', 'Dubai, UAE']:
                risk_factors.append("High-risk location detected")
            else:
                risk_factors.append("Unusual transaction location")
        
        # Device risk (more detailed)
        if transaction_data['device'] != transaction_data['usual_device']:
            if transaction_data['device'] == 'other':
                risk_factors.append("Unknown device type used")
            else:
                risk_factors.append("Unusual device used")
        
        # Time risk (more detailed)
        transaction_time = datetime.strptime(transaction_data['time'], '%Y-%m-%dT%H:%M')
        hour = transaction_time.hour
        
        if transaction_data['usual_time'] == 'morning' and (hour < 6 or hour >= 12):
            risk_factors.append("Transaction outside usual morning hours")
        elif transaction_data['usual_time'] == 'afternoon' and (hour < 12 or hour >= 18):
            risk_factors.append("Transaction outside usual afternoon hours")
        elif transaction_data['usual_time'] == 'evening' and (hour < 18 or hour >= 24):
            risk_factors.append("Transaction outside usual evening hours")
        elif transaction_data['usual_time'] == 'night' and (hour >= 6):
            risk_factors.append("Transaction outside usual night hours")
        
        return risk_factors

def create_sample_data():
    """Create sample data for testing"""
    sample_data = []
    
    # Create normal user transactions
    normal_user = [
        {'amount': 100, 'time': datetime.now().strftime('%Y-%m-%dT%H:%M'),
         'location': 'Los Angeles, USA', 'device': 'mobile',
         'usual_time': 'morning', 'usual_location': 'Los Angeles, USA',
         'usual_device': 'mobile', 'is_fraud': 0},
        {'amount': 150, 'time': datetime.now().strftime('%Y-%m-%dT%H:%M'),
         'location': 'Los Angeles, USA', 'device': 'mobile',
         'usual_time': 'morning', 'usual_location': 'Los Angeles, USA',
         'usual_device': 'mobile', 'is_fraud': 0},
        {'amount': 200, 'time': datetime.now().strftime('%Y-%m-%dT%H:%M'),
         'location': 'Los Angeles, USA', 'device': 'mobile',
         'usual_time': 'morning', 'usual_location': 'Los Angeles, USA',
         'usual_device': 'mobile', 'is_fraud': 0}
    ]
    sample_data.append(normal_user)
    
    # Create fraudulent user transactions
    fraud_user = [
        {'amount': 100, 'time': datetime.now().strftime('%Y-%m-%dT%H:%M'),
         'location': 'Los Angeles, USA', 'device': 'mobile',
         'usual_time': 'morning', 'usual_location': 'Los Angeles, USA',
         'usual_device': 'mobile', 'is_fraud': 0},
        {'amount': 150, 'time': datetime.now().strftime('%Y-%m-%dT%H:%M'),
         'location': 'Los Angeles, USA', 'device': 'mobile',
         'usual_time': 'morning', 'usual_location': 'Los Angeles, USA',
         'usual_device': 'mobile', 'is_fraud': 0},
        {'amount': 5000, 'time': datetime.now().strftime('%Y-%m-%dT%H:%M'),
         'location': 'Moscow, Russia', 'device': 'other',
         'usual_time': 'morning', 'usual_location': 'Los Angeles, USA',
         'usual_device': 'mobile', 'is_fraud': 1}
    ]
    sample_data.append(fraud_user)
    
    return sample_data

def test_model():
    """Test the model with sample transactions"""
    model = FraudDetectionModel()
    
    # Create and train the model
    sample_data = create_sample_data()
    model.train(sample_data)
    
    # Test normal transaction
    normal_transaction = {
        'amount': 100,
        'time': datetime.now().strftime('%Y-%m-%dT%H:%M'),
        'location': 'Los Angeles, USA',
        'device': 'mobile',
        'usual_time': 'morning',
        'usual_location': 'Los Angeles, USA',
        'usual_device': 'mobile'
    }
    
    # Test suspicious transaction
    suspicious_transaction = {
        'amount': 2000,
        'time': datetime.now().strftime('%Y-%m-%dT%H:%M'),
        'location': 'Moscow, Russia',
        'device': 'other',
        'usual_time': 'morning',
        'usual_location': 'Los Angeles, USA',
        'usual_device': 'mobile'
    }
    
    # Test predictions
    normal_prob = model.predict(normal_transaction)
    suspicious_prob = model.predict(suspicious_transaction)
    
    print("Normal Transaction:")
    print(f"Prediction: {'Fraudulent' if normal_prob > 0.5 else 'Not Fraudulent'}")
    print(f"Confidence: {normal_prob:.2%}")
    print("Risk Factors:", model.get_risk_factors(normal_transaction))
    
    print("\nSuspicious Transaction:")
    print(f"Prediction: {'Fraudulent' if suspicious_prob > 0.5 else 'Not Fraudulent'}")
    print(f"Confidence: {suspicious_prob:.2%}")
    print("Risk Factors:", model.get_risk_factors(suspicious_transaction))

if __name__ == "__main__":
    test_model() 