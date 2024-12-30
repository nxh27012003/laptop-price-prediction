import streamlit as st
import pickle as pk
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)
from predict_price import *
from rf_predict_price import *

def gui_app():
    st.title("Laptop Price Prediction")
    st.caption("Introduction to Data Science")

    # Brand & OS
    BRAND_OPTIONS = ["Apple", "Dell", "Lenovo", "HP", "Asus", "Acer", "MSI", "Microsoft", "Other"]
    OS_OPTIONS = ["macOS", "Windows 11", "Windows 11 Home", "Windows 11 Pro", "Windows 10", "Chrome OS", "Other"]
    brd_col, ops_col = st.columns(2)
    with brd_col:
        brand_input = st.selectbox(label="Brand", options=BRAND_OPTIONS)
        if brand_input=="Other":
            brand_input = st.text_input("Brand")
            if brand_input == "":
                brand_input = "Dell"
    
    with ops_col:
        os_input = st.selectbox(label="Operating System", options=OS_OPTIONS)
        if os_input=="Other":
            os_input = st.text_input("Operating System")
            if os_input == "":
                os_input = "Windows 11"

    # CPU & GPU Input
    cpu_input = st.text_input(label="CPU", placeholder="e.g. Intel Core i7 12700H..")
    gpu_input = st.text_input(label="GPU", placeholder="e.g. Intel Iris Xe..")

    # RAM
    RAM_OPTIONS = ["4GB", "8GB", "16GB", "32GB",  "64GB", "128GB", "256GB", "Other"]
    STORAGE_OPTIONS = ["64GB", "128GB", "256GB", "512GB", "1TB", "2TB", "4TB", "Other"]
    ram_col, stg_col = st.columns(2)
    with ram_col:
        ram_input = st.selectbox("RAM", options=RAM_OPTIONS)
        if ram_input=="Other":
            rcol1, rcol2 = st.columns([1,3])
            with rcol1:
                rtype = st.selectbox("Unit", options=["GB", "TB"])
            with rcol2:
                rvalue = st.text_input("Enter RAM value", placeholder="e.g. 24")
                if rvalue == "":
                    rvalue = "2"
            ram_input = str(rvalue) + rtype
    
    with stg_col:
        storage_input = st.selectbox("Storage", options=STORAGE_OPTIONS)
        if storage_input=="Other":
            scol1, scol2 = st.columns([1,3])
            with scol1:
                stype = st.selectbox("Unit", options=["GB", "TB"])
            with scol2:
                svalue = st.text_input("Enter Storage value", placeholder="e.g. 192")
                if svalue == "":
                    svalue == "2"

            storage_input = str(svalue) + stype

    # Screen Size
    SCREEN_SIZE_OPTIONS = ["1366x768", "1600x900", "1920x1080", "1920x1200", "2560x1440", "2560x1600", "3024x1964", "3072x1920", "3840x2160", "3840x2400", "Other"]
    resolution_input = st.selectbox("Resolution", options=SCREEN_SIZE_OPTIONS)
    if resolution_input == "Other":
        wcol, hcol = st.columns(2)
        with wcol:
            width = st.text_input("Width (pixels)", placeholder="e.g. 1920")
            if width == "":
                width = 1920
        with hcol:
            height = st.text_input("Height (pixels)", placeholder="e.g. 1080")
            if height == "":
                height = 1080
        resolution_input = str(width) + "x" + str(height)
    
    # Monitor 
    MONITOR_OPTIONS = [x / 10 for x in range(106, 200)]
    monitor_input = st.select_slider("Monitor", options=MONITOR_OPTIONS)

    WEIGHT_RANGE = [x / 100 for x in range(32, 860)]
    weight_input = st.select_slider("Weight", WEIGHT_RANGE)

    # Display selected values
    st.write("___")
    st.write("Features Summary:")
    if cpu_input == "":
        cpu_input = "Intel Core i7 12700H"
    if gpu_input == "":
        gpu_input = "Intel Iris Xe"
    st.table({
        "Brand": brand_input,
        "Operating System": os_input,
        "CPU": cpu_input,
        "GPU": gpu_input,
        "RAM": ram_input,
        "Storage": storage_input,
        "Monitor": str(monitor_input) + "\"",
        "Resolution": resolution_input,
        "Weight": str(weight_input) + "kg"
    })

    st.write("___")

    model_name = st.radio("Model", options=["Random Forest", "K-Nearest Neighbor"])

    if st.button('Submit'):
        df = transfer_to_df(brand=brand_input,
                                   cpu=cpu_input,
                                   gpu=gpu_input,
                                   monitor=str(monitor_input),
                                   resolution=resolution_input,
                                   ram=ram_input,
                                   storage=storage_input,
                                   os=os_input,
                                   weight=str(weight_input))
        # st.dataframe(df)
        
        MODELNAME = "knn_model_2.pkl"
        MODEL_PATH = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__)), MODELNAME))

        SCALERNAME = "save_scaler.pkl"
        SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), SCALERNAME))
        
        if model_name == 'K-Nearest Neighbor':
            price = knn_predict_price(df, MODEL_PATH, SCALER_PATH)

        if model_name == 'Random Forest':
            price = rf_predict_price(brand_input, cpu_input, gpu_input, str(monitor_input), resolution_input, ram_input, storage_input, os_input, str(weight_input))
        st.success(f'{price:.2f} USD')

    else:
        st.success("0 USD")

if __name__ == '__main__':
    gui_app()