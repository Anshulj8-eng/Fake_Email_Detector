"""
Train a supervised learning model for fake email detection
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def load_data(filepath):
    """Load the dataset"""
    df = pd.read_csv(filepath)
    return df


def data_preprocessing(df):
    feature_columns = [
        'has_link',
            'has_money_words',
            'has_urgent_words',
            'has_caps',
            'num_exclamation',
            'email_length',
            'num_digits',
            'num_special_chars'
    ]

    X = df[feature_columns]
    y = df['is_spam']
    return X,y,feature_columns

def train_model(X_train,y_train):
    """Train Random Forest classifier"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model,X_test,y_test):
    """Evaluate the trained model"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, conf_matrix

def save_model(model, scaler, feature_columns, model_dir):
    """Save model and scaler"""
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'fake_email_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(feature_columns, os.path.join(model_dir, 'feature_columns.pkl'))
    
    print(f"Model saved to {model_dir}")

def main():
    # Get the directory of this script
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'spam_email.csv')
    model_dir = os.path.join(base_dir, 'models')
    
    # Check if dataset exists, if not generate it
    if not os.path.exists(data_path):
        print("Dataset not found. Generating...")
        import sys
        sys.path.append(os.path.join(base_dir, 'data'))
        from generate_dataset import generate_dataset
        df = generate_dataset(1000)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print("Dataset generated!")
    

    # Load data
    print("Loading data...")
    df = load_data(data_path)
    
    # Preprocess
    print("Preprocessing data...")
    X, y, feature_columns = data_preprocessing(df)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)

    
    # Evaluate
    print("Evaluating model...")
    accuracy, report, conf_matrix = evaluate_model(model, X_test, y_test)
    
    print(f"\n{'='*50}")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Save model
    save_model(model, scaler, feature_columns, model_dir)
    
    # Feature importance
    print("\nFeature Importance:")
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.to_string(index=False))

if __name__ == "__main__":
    main()
