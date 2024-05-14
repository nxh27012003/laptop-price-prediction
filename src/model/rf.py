from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

def train_rf(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """
    Train a Random Forest regressor model.

    Parameters:
    X_train (pd.DataFrame): The training feature set.
    y_train (pd.Series): The training target set.
    n_estimators (int): The number of trees in the forest.
    max_depth (int): The maximum depth of the tree.
    random_state (int): Random seed for reproducibility.

    Returns:
    model (RandomForestRegressor): The trained Random Forest model.
    scaler (StandardScaler): The scaler used to standardize the data.
    """
    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initializing and training the Random Forest regressor
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf.fit(X_train_scaled, y_train)

    return rf, scaler

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from data.divide_data import split_data

    df = pd.read_csv("processed_final_dataset.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_column="Price")

    rf_model, scaler = train_rf(X_train, y_train)
    
    # Saving the model and scaler to disk
    with open('rf_model.pkl', 'wb') as model_file:
        pickle.dump(rf_model, model_file)
    with open('rf_scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    print("Random Forest model and scaler saved successfully.")
