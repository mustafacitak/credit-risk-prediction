import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import yaml
import json
from datetime import datetime


def load_processed_data():
    """Load processed training data"""
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    return X_train, y_train.values.ravel()


def train_model(X_train, y_train):
    """Train a Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


def save_model_and_metrics(model, X_train, y_train):
    """Save the trained model and training metrics"""
    # Save model
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/model_{timestamp}.joblib'
    joblib.dump(model, model_path)

    # Calculate and save training metrics
    train_score = model.score(X_train, y_train)
    feature_importance = dict(zip(X_train.columns, model.feature_importances_))

    metrics = {
        'timestamp': timestamp,
        'training_accuracy': train_score,
        'model_path': model_path,
        'top_features': dict(sorted(feature_importance.items(),
                                    key=lambda x: x[1],
                                    reverse=True)[:5])
    }

    with open(f'models/metrics_{timestamp}.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics


def main():
    # Load data
    X_train, y_train = load_processed_data()

    # Train model
    model = train_model(X_train, y_train)

    # Save model and metrics
    metrics = save_model_and_metrics(model, X_train, y_train)
    print(f"Training completed. Metrics: {metrics}")


if __name__ == "__main__":
    main()