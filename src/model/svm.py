from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pickle

def train_svm(X_train, y_train, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
    """
    Train a Support Vector Machine regressor model.

    Parameters:
    X_train (pd.DataFrame): The training feature set.
    y_train (pd.Series): The training target set.
    kernel (str): Specifies the kernel type to be used in the algorithm.
    C (float): Regularization parameter.
    epsilon (float): Epsilon in the epsilon-SVR model.
    gamma (str): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.

    Returns:
    model (SVR): The trained SVM model.
    scaler (StandardScaler): The scaler used to standardize the data.
    """
    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initializing and training the SVM regressor
    svm = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
    svm.fit(X_train_scaled, y_train)

    return svm, scaler

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from data.divide_data import split_data

    df = pd.read_csv("processed_final_dataset.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_column="Price")

    svm_model, scaler = train_svm(X_train, y_train)
    
    # Saving the model and scaler to disk
    with open('svm_model.pkl', 'wb') as model_file:
        pickle.dump(svm_model, model_file)
    with open('svm_scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    print("SVM model and scaler saved successfully.")
