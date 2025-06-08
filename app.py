from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join('model', "C:/Users/SAIF/Downloads/email_classification/model/email_classification_model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load the vocabulary (3000 most common words)
vocab_path = os.path.join('model', "C:/Users/SAIF/Downloads/email_classification/model/emails.csv")
df_vocab = pd.read_csv(vocab_path)
vocab = df_vocab.columns[1:-1]  # Exclude 'Email No.' and 'Prediction'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the email text from the form
        email_text = request.form['email']
        
        # Preprocess the email (same as in the notebook)
        email_words = email_text.lower().split()
        
        # Create a feature vector with counts of each vocabulary word
        features = np.zeros(len(vocab))
        for word in email_words:
            if word in vocab:
                idx = np.where(vocab == word)[0][0]
                features[idx] += 1
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0]
        
        # Prepare response
        result = {
            'prediction': 'Spam' if prediction[0] == 1 else 'Not Spam',
            'spam_probability': float(probability[1]),
            'ham_probability': float(probability[0])
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)