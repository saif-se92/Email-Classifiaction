![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-orange.svg)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression-green.svg)


# Email Spam Classification Flask App

This is a Flask web application that classifies emails as spam or not spam using a Logistic Regression model trained on the Email Spam Classification Dataset from Kaggle.

## Features

- Web interface for entering email text
- Real-time classification results
- Probability scores for both spam and not spam classifications
- Responsive design

## Dataset

The model was trained on the [Email Spam Classification Dataset CSV](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv) from Kaggle. The dataset contains 5172 emails with 3000 features representing the count of the most common words in each email.

## Model

The application uses a Logistic Regression model with the following performance metrics:
- Accuracy: ~97.3%
- Confusion Matrix:
  - True Negative: 721
  - False Positive: 14
  - False Negative: 14
  - True Positive: 286

## Installation

1. Clone this repository:
   ```bash
     git clone https://github.com/yourusername/email_spam_classifier.git
     cd email_spam_classifier
2. Create and activate a virtual environment:
   ```bash
     python -m venv venv
     source venv/bin/activate
3. Install the required packages:
    ```bash
      pip install -r requirements.txt
4. Download the model and dataset:

    Place the trained model (logistic_regression_model.pkl) in the model/ directory

    Place the dataset (emails.csv) in the model/ directory
## Usage:
Run the Flask application:
  ```bash
    python app.py
