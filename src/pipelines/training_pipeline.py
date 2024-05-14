import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

from data.ingest_data import load_data
from data.preprocessing import preprocess_data
from data.divide_data import split_data
from model.knn import train_knn
from model.svm import train_svm
from model.rf import train_rf

def train_model(X_train, y_train, model_type: str):
    """
    Train a specified model type.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    model_type (str): The type of model to train ('knn', 'svm', 'rf').

    Returns:
    model: The trained model.
    scaler: The scaler used for the data.
    """
    if model_type == 'knn':
        return train_knn(X_train, y_train)
    elif model_type == 'svm':
        return train_svm(X_train, y_train)
    elif model_type == 'rf':
        return train_rf(X_train, y_train)
    else:
        raise ValueError("Invalid model type specified. Choose from 'knn', 'svm', 'rf'.")

def save_model_and_scaler(model, scaler, model_path: str, scaler_path: str):
    """
    Save the model and scaler to disk.

    Parameters:
    model: The trained model.
    scaler: The scaler used for the data.
    model_path (str): Path to save the model.
    scaler_path (str): Path to save the scaler.
    """
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(scaler_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data("../../../data/processed/final_dataset.csv")
    df_cleaned = preprocess_data(df)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_cleaned, target_column="Price")

    # Train models
    knn_model, knn_scaler = train_model(X_train, y_train, model_type='knn')
    svm_model, svm_scaler = train_model(X_train, y_train, model_type='svm')
    rf_model, rf_scaler = train_model(X_train, y_train, model_type='rf')

    # Save models and scalers
    save_model_and_scaler(knn_model, knn_scaler, 'knn_model.pkl', 'knn_scaler.pkl')
    save_model_and_scaler(svm_model, svm_scaler, 'svm_model.pkl', 'svm_scaler.pkl')
    save_model_and_scaler(rf_model, rf_scaler, 'rf_model.pkl', 'rf_scaler.pkl')

    print("Models and scalers saved successfully.")
