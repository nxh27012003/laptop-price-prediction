import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, val_size: float = 0.1, random_seed: int = 42):
    """
    Split the data into training, validation, and test sets.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    test_size (float): The proportion of the data to include in the test set.
    val_size (float): The proportion of the training data to include in the validation set.
    random_seed (int): The random seed for reproducibility.

    Returns:
    tuple: A tuple containing the training, validation, and test sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    # Split the training data into training and validation sets
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size proportion to the training set size
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size_adjusted, random_state=random_seed)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("processed_final_dataset.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_column="Price")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
