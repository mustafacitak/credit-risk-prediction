import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import os
from utils import plot_feature_importance, plot_confusion_matrix



def load_test_data():
    """İşlenmiş test verisini yükler"""
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    return X_test, y_test.values.ravel()


def load_model(model_path):
    """Kaydedilmiş modeli yükler"""
    return joblib.load(model_path)


def evaluate_model(model, X_test, y_test):
    """Modeli test verisi üzerinde değerlendirir ve metrikleri döner"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest Doğruluğu: {accuracy:.2f}")
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))
    print("\nKarmaşıklık Matrisi:")
    print(cm)

    return accuracy, report, cm


def save_metrics(accuracy, report, cm):
    """Test metriklerini kaydeder"""
    os.makedirs('models/metrics', exist_ok=True)
    metrics = {
        'test_accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

    with open('models/metrics/test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("\nTest metrikleri kaydedildi.")


def main():
    # Test verisini yükleme
    X_test, y_test = load_test_data()

    # Eğitilmiş modeli yükleme
    model_path = 'models/model_20241024_165933.joblib'  # En son kaydettiğiniz modelin yolunu girin
    model = load_model(model_path)

    # Modeli değerlendirme
    accuracy, report, cm = evaluate_model(model, X_test, y_test)

    # Metrikleri kaydetme
    save_metrics(accuracy, report, cm)

    # Karmaşıklık matrisini görselleştirme ve gösterme
    plot_confusion_matrix(cm)



if __name__ == "__main__":
    main()
