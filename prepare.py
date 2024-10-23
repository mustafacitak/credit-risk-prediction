import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import os


def load_data(filepath):
    """Load and perform initial cleaning of credit data"""
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
    """Preprocess the credit data"""
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))

    # Convert categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_columns)

    return df


def prepare_features_target(df, target_column):
    """Separate features and target, scale features"""
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y


def split_and_save_data(X, y, test_size=0.2, random_state=42):
    """Split data and save to processed directory"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Save processed data
    os.makedirs('data/processed', exist_ok=True)

    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    # Save feature names for later use
    with open('data/processed/feature_names.yaml', 'w') as f:
        yaml.dump({'feature_names': X_train.columns.tolist()}, f)


def main():
    # Load raw data
    df = load_data('data/raw/credit_data.csv')

    # Preprocess
    df_processed = preprocess_data(df)

    # Prepare features and target
    X, y = prepare_features_target(df_processed, target_column='default')

    # Split and save
    split_and_save_data(X, y)


if __name__ == "__main__":
    main()