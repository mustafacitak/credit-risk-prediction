import os
import yaml
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Proje yapılandırma sabitleri
CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'model_params': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'class_weight': 'balanced'
    },
    'categorical_columns': ['SEX', 'EDUCATION', 'MARRIAGE'],
    'target_column': 'default.payment.next.month'
}

def get_project_root():
    """Proje kök dizinini döndürür"""
    return Path(__file__).parent.parent

def setup_directories():
    """Gerekli dizin yapısını oluşturur"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'models/metrics'
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

def load_yaml(filepath):
    """YAML dosyasını yükler"""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, filepath):
    """Veriyi YAML formatında kaydeder"""
    with open(filepath, 'w') as f:
        yaml.dump(data, f)

def save_json(data, filepath):
    """Veriyi JSON formatında kaydeder"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def get_latest_model_path():
    """En son eğitilen model dosyasının yolunu döndürür"""
    model_dir = os.path.join(get_project_root(), 'models')
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    if not model_files:
        raise FileNotFoundError("Hiç model dosyası bulunamadı.")
    latest_model = sorted(model_files)[-1]
    return os.path.join(model_dir, latest_model)

def plot_feature_importance(model, feature_names, top_n=10):
    """Özellik önemlilik grafiğini oluşturur ve kaydeder"""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance)
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()

    # Grafiği kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'models/feature_importance_{timestamp}.png')
    plt.close()

def plot_confusion_matrix(cm, save=True):
    """Karmaşıklık matrisini görselleştirir"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'models/confusion_matrix_{timestamp}.png')
    plt.close()

def log_training_info(metrics, model_path):
    """Eğitim bilgilerini loglar"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(get_project_root(), 'models', 'training_log.txt')

    with open(log_file, 'a') as f:
        f.write(f"\n=== Training Session: {timestamp} ===\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Training Accuracy: {metrics['training_accuracy']:.4f}\n")
        f.write("Top Features:\n")
        for feature, importance in metrics['top_features'].items():
            f.write(f"  - {feature}: {importance:.4f}\n")
        f.write("=" * 50 + "\n")
