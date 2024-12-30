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

import warnings
warnings.filterwarnings("ignore")

## TRAIN MODEL
def train_model(X_train, y_train, model_name: str, grid_search=False):
    if model_name=='LinearSVR':
        if grid_search==True:
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100, 200],
                'epsilon': [0.01, 0.1, 1, 10],
                'max_iter': [100, 500, 800, 1000],
                'tol': [1e-4, 1e-3, 1e-2]
            }

            svr = LinearSVR()
            grid_search = GridSearchCV(svr, param_grid, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)

            print("Best Params:", grid_search.best_params_)
            print("Best Score:", -grid_search.best_score_)

            return grid_search.best_estimator_
        
        else:
            svr = LinearSVR()
            svr.fit(X_train, y_train)

            return svr
    
    if model_name=='KNN':
        if grid_search==True:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }

            knn = KNeighborsRegressor()
            grid_search = GridSearchCV(knn, param_grid, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)

            print("Best Params:", grid_search.best_params_)
            print("Best Score:", -grid_search.best_score_)

            return grid_search.best_estimator_
        
        else:
            knn = KNeighborsRegressor()
            knn.fit(X_train, y_train)

            return knn

## PREDICT IN THE X_TEST
def predict(model: LinearSVR, X_test, y_test):
    y_pred = model.predict(X_test)
    print(mean_squared_error(y_test, y_pred))
    
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    print(df)

## EVALUATION ON TEST DATA
def evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Mean Absolute Percentage Error: {mape:.2f}%')

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

## SAVE MODEL
def save_model(model, model_name, scaler=None):
    model_filename = f"{model_name}_model.pkl"
    joblib.dump(model, model_filename)

    # Save the scaler if provided
    if scaler:
        scaler_filename = f"{model_name}_scaler.pkl"
        joblib.dump(scaler, scaler_filename)

    print(f"Model saved as {model_filename}")
