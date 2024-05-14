from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import pickle

def train_knn(X_train, y_train, n_neighbors=5, weights='uniform', algorithm='auto'):
    """
    Train a K-Nearest Neighbors regressor model.

    Parameters:
    X_train (pd.DataFrame): The training feature set.
    y_train (pd.Series): The training target set.
    n_neighbors (int): Number of neighbors to use.
    weights (str): Weight function used in prediction.
    algorithm (str): Algorithm used to compute the nearest neighbors.

    Returns:
    model (KNeighborsRegressor): The trained KNN model.
    scaler (StandardScaler): The scaler used to standardize the data.
    """
    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initializing and training the KNN regressor
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    knn.fit(X_train_scaled, y_train)

    return knn, scaler

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from data.divide_data import split_data

    df = pd.read_csv("processed_final_dataset.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_column="Price")

    knn_model, scaler = train_knn(X_train, y_train)
    
    # Saving the model and scaler to disk
    with open('knn_model.pkl', 'wb') as model_file:
        pickle.dump(knn_model, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    print("KNN model and scaler saved successfully.")
