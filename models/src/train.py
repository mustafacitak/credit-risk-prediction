import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import yaml
import json
from datetime import datetime
from utils import plot_feature_importance, plot_confusion_matrix


def setup_visualization_directory():
    """Görselleştirme için gerekli dizinleri oluşturur"""
    vis_dir = os.path.join(get_project_root(), 'models')
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir

def load_processed_data():
    """ İşlenmiş eğitim verisini yükler """
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    return X_train, y_train.values.ravel()


def train_model(X_train, y_train):
    """ Random Forest modelini eğitir """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


def save_model_and_metrics(model, X_train, y_train):
    """ Eğitilen modeli ve eğitim metriklerini kaydeder """
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/model_{timestamp}.joblib'
    joblib.dump(model, model_path)

    # Eğitim doğruluğunu hesaplama
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

    print(f"Model ve metrikler kaydedildi. Eğitim doğruluğu: {train_score:.2f}")


def main():
    # İşlenmiş veriyi yükleme
    X_train, y_train = load_processed_data()

    # Modeli eğitme
    model = train_model(X_train, y_train)

    # Modeli ve metrikleri kaydetme
    save_model_and_metrics(model, X_train, y_train)

    # Özellik önemliliği grafiğini oluşturma ve gösterme
    plot_feature_importance(model, X_train.columns)



if __name__ == "__main__":
    main()