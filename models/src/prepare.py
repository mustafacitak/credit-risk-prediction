import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import os


def load_data(filepath):
    """Load and perform initial cleaning of credit data"""
    df = pd.read_csv(filepath)
    print("\nVeri Yüklendi - İlk 5 Satır:")
    print(df.head())
    return df


def preprocess_data(df):
    """Preprocess the credit data"""
    # Gereksiz kolonları kaldır (ID kolonu)
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        print("\n'ID' Kolonu Kaldırıldı - İlk 5 Satır:")
        print(df.head())

    # Handle missing numeric values
    df = df.fillna(df.mean(numeric_only=True))
    print("\nEksik Değerler Dolduruldu - İlk 5 Satır:")
    print(df.head())

    # Kategorik değişkenleri belirle ve one-hot encoding uygula
    categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE']
    df = pd.get_dummies(df, columns=categorical_columns)
    print("\nKategorik Değişkenler One-Hot Encoding ile Dönüştürüldü - İlk 5 Satır:")
    print(df.head())

    return df


def prepare_features_target(df, target_column):
    """Separate features and target, scale features"""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    print("\nÖzellikler ve Hedef Ayrıldı:")
    print("X (İlk 5 Satır):")
    print(X.head())
    print("\nY (İlk 5 Değer):")
    print(y.head())

    # Özellikleri ölçeklendirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print("\nÖzellikler Ölçeklendirildi - İlk 5 Satır:")
    print(X_scaled.head())

    return X_scaled, y


def split_and_save_data(X, y, test_size=0.2, random_state=42):
    """Split data and save to processed directory"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("\nEğitim ve Test Setleri Ayrıldı:")
    print(f"X_train (İlk 5 Satır):\n{X_train.head()}")
    print(f"y_train (İlk 5 Değer):\n{y_train.head()}")
    print(f"X_test (İlk 5 Satır):\n{X_test.head()}")
    print(f"y_test (İlk 5 Değer):\n{y_test.head()}")

    # İşlenmiş veriyi kaydetme
    os.makedirs('data/processed', exist_ok=True)

    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    # Özellik adlarını kaydetme
    with open('data/processed/feature_names.yaml', 'w') as f:
        yaml.dump({'feature_names': X_train.columns.tolist()}, f)

    print("\nVeriler Kaydedildi.")


def main():
    # Veri dosyasını yükleme
    filepath = '/Users/mustafacitak/Documents/GitHub/credit-risk-prediction/data/raw/credit_data.csv'
    df = load_data(filepath)

    # Veri ön işleme
    df_processed = preprocess_data(df)

    # Özellikleri ve hedef değişkeni ayırma
    X, y = prepare_features_target(df_processed, target_column='default.payment.next.month')

    # Veriyi eğitim ve test setlerine ayırma ve kaydetme
    split_and_save_data(X, y)


if __name__ == "__main__":
    main()
