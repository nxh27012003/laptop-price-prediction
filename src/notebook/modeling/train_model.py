import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import joblib
from processing import *
from model_utils import *

import warnings
warnings.filterwarnings("ignore")


if __name__=='__main__':
    DATA_FILENAME = "final_dataset.csv"
    PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    PATH_TO_DATA = os.path.abspath(os.path.join(PARENT_DIR, "..", "..", "data", "processed", DATA_FILENAME))

    df = process_data(PATH_TO_DATA)

    df1 = encoding_data(df, columns=["Brand", "OS", "CPU Brand", "GPU Brand"])
    target_corr = df1.corr()["Price"].apply(abs).sort_values()

    SELECTED_FEATURES = 17
    selected_features = list(target_corr[-SELECTED_FEATURES:].index)

    limited_df = df1[selected_features]

    X, y = limited_df.drop("Price", axis=1), limited_df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

    columns_to_scale = ["Weight", "Monitor", "RAM", "Storage Amount", "GPU Mark", "Width", "Height", "CPU Mark"]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train_scaled[columns_to_scale])

    X_test_scaled = X_test.copy()
    X_test_scaled[columns_to_scale] = scaler.fit_transform(X_test_scaled[columns_to_scale])

    MODEL_NAME = 'KNN'
    GRID_SEARCH = False

    trained_model = train_model(X_train=X_train_scaled, y_train=y_train, model_name=MODEL_NAME, grid_search=GRID_SEARCH)

    predict(model=trained_model, X_test=X_test_scaled, y_test=y_test)
    evaluation(model=trained_model, X_test=X_test_scaled, y_test=y_test)
    save_model(model=train_model, model_name="knn_default", scaler=scaler)