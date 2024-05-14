import pandas as pd
import pickle

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

def preprocess_features(df: pd.DataFrame):
    """
    Preprocess the input features for prediction.

    Parameters:
    df (pd.DataFrame): The input feature DataFrame.

    Returns:
    df (pd.DataFrame): The preprocessed feature DataFrame.
    """
    # This should match the preprocessing steps used in preprocessing_data.py
    # For the sake of this example, assuming it involves dummy encoding
    df = pd.get_dummies(df)
    
    # Add any missing columns that might be present in the training set but not in the input
    expected_columns = ['Monitor', 'Width', 'RAM', 'Storage Amount', 'Weight', 'L3-Cache',
                        'Brand_apple', 'Brand_dell', 'Brand_hp', 'Brand_lenovo', 'Brand_other', 
                        'OS_Chrome OS', 'OS_MacOS', 'OS_Ubuntu', 'OS_Windows 10', 'OS_Windows 8',
                        'CPU Brand_amd', 'CPU Brand_intel', 'GPU Brand_geforce', 'GPU Brand_radeon']
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Ensure the columns are in the same order as the training set
    df = df[expected_columns]
    
    return df

def predict_price(model_path: str, scaler_path: str, features: pd.DataFrame):
    """
    Predict the price of a laptop based on input features.

    Parameters:
    model_path (str): Path to the saved model.
    scaler_path (str): Path to the saved scaler.
    features (pd.DataFrame): Input features for the prediction.

    Returns:
    float: The predicted price.
    """
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    features_preprocessed = preprocess_features(features)
    features_scaled = scaler.transform(features_preprocessed)
    predicted_price = model.predict(features_scaled)
    return predicted_price

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Monitor': [15.6],
        'Width': [360],
        'RAM': [8],
        'Storage Amount': [512],
        'Weight': [2.5],
        'L3-Cache': [6],
        'Brand': ['dell'],
        'OS': ['Windows 10'],
        'CPU Brand': ['intel'],
        'GPU Brand': ['geforce']
    }
    sample_df = pd.DataFrame(sample_data)
    predicted_price = predict_price('rf_model.pkl', 'rf_scaler.pkl', sample_df)
    print(f"Predicted Price: {predicted_price[0]}")
