"""
Model training and evaluation for stress detection.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def prepare_dataset(features, labels):
    """
    Prepare dataset for training.
    
    Args:
        features: Feature array (n_samples, n_features)
        labels: Label array (n_samples,)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(features, labels, test_size=0.2, random_state=42)


def train_stress_model(X_train, y_train, model_type='rf'):
    """
    Train stress detection model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Model type ('rf' for Random Forest)
        
    Returns:
        Trained model
    """
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'accuracy': model.score(X_test, y_test),
        'predictions': y_pred
    }
