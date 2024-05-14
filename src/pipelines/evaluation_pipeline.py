import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from data.preprocessing import preprocess_data
from data.divide_data import split_data

def load_model_and_scaler(model_path: str, scaler_path: str):
    """
    Load the model and scaler from disk.

    Parameters:
    model_path (str): Path to the saved model.
    scaler_path (str): Path to the saved scaler.

    Returns:
    model: The loaded model.
    scaler: The loaded scaler.
    """
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    """
    Evaluate the model on the test set.

    Parameters:
    model: The trained model.
    scaler: The scaler used for the data.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target.

    Returns:
    dict: A dictionary containing the evaluation metrics.
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'mean_squared_error': mse, 'r2_score': r2}

if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_csv("processed_final_dataset.csv")
    df_cleaned = preprocess_data(df)

    # Split data
    _, _, X_test, _, _, y_test = split_data(df_cleaned, target_column="Price")

    # Evaluate KNN model
    knn_model, knn_scaler = load_model_and_scaler('knn_model.pkl', 'knn_scaler.pkl')
    knn_eval = evaluate_model(knn_model, knn_scaler, X_test, y_test)
    
    # Evaluate SVM model
    svm_model, svm_scaler = load_model_and_scaler('svm_model.pkl', 'svm_scaler.pkl')
    svm_eval = evaluate_model(svm_model, svm_scaler, X_test, y_test)
    
    # Evaluate RF model
    rf_model, rf_scaler = load_model_and_scaler('rf_model.pkl', 'rf_scaler.pkl')
    rf_eval = evaluate_model(rf_model, rf_scaler, X_test, y_test)

    # Save evaluation results to CSV
    eval_results = pd.DataFrame({
        'Model': ['KNN', 'SVM', 'Random Forest'],
        'Mean Squared Error': [knn_eval['mean_squared_error'], svm_eval['mean_squared_error'], rf_eval['mean_squared_error']],
        'R2 Score': [knn_eval['r2_score'], svm_eval['r2_score'], rf_eval['r2_score']]
    })
    eval_results.to_csv('model_evaluation_results.csv', index=False)

    print("Evaluation results saved successfully.")
