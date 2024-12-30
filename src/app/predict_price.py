import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from map_cpu_gpu import *


def transfer_to_df(brand,
                   cpu, 
                   gpu, 
                   monitor, 
                   resolution, 
                   ram, 
                   storage, 
                   os, 
                   weight) -> pd.DataFrame:
    
    df = pd.DataFrame(columns=['Brand_hp', 'GPU Brand_rtx', 'GPU Brand_nvidia', 'Weight', 'Monitor',
                               'GPU Brand_intel', 'GPU Brand_geforce', 'OS_MacOS', 'Brand_apple',
                               'CPU Brand_apple', 'RAM', 'Storage Amount', 'GPU Mark', 'Width',
                               'Height', 'CPU Mark'])

    # Brand_hp & Brand_apple
    if brand.lower() == "apple":
        df.at[0, "Brand_hp"] = 0
        df.at[0, "Brand_apple"] = 1
    if brand.lower() == "hp":
        df.at[0, "Brand_hp"] = 1
        df.at[0, "Brand_apple"] = 0
    if brand.lower() != "apple" and brand.lower() != "hp":
        df.at[0, "Brand_hp"] = 0
        df.at[0, "Brand_apple"] = 0

    # OS_MacOS
    if os.lower() == "macos":
        df.at[0, "OS_MacOS"] = 1
    if os.lower() != "macos":
        df.at[0, "OS_MacOS"] = 0


    # GPU_rtx & GPU Brand_nvidia & GPU Brand_intel & GPU Brand_geforce
    df.at[0, "GPU Brand_rtx"] = 0
    df.at[0, "GPU Brand_nvidia"] = 0
    df.at[0, "GPU Brand_intel"] = 0
    df.at[0, "GPU Brand_geforce"] = 0
    if "intel" in gpu.lower():
        df.at[0, "GPU Brand_intel"] = 1
    if "nvidia" in gpu.lower():
        df.at[0, "GPU Brand_nvidia"] = 1
    if "rtx" in gpu.lower():
        df.at[0, "GPU Brand_rtx"] = 1
    if "geforce" in gpu.lower():
        df.at[0, "GPU Brand_geforce"] = 1


    if "apple" in cpu.lower():
        df.at[0, "CPU Brand_apple"] = 1
    if "apple" not in cpu.lower():
        df.at[0, "CPU Brand_apple"] = 0

    df.at[0, "Weight"] = float(weight)
    df.at[0, "Monitor"] = float(monitor)
    
    if ram[-2:] == "TB":
        df.at[0, "RAM"] = float(ram[:-2]) * 1024
    else:
        df.at[0, "RAM"] = float(ram[:-2])

    if storage[-2:] == "TB":
        df.at[0, "Storage Amount"] = float(storage[:-2]) * 1024
    else:
        df.at[0, "Storage Amount"] = float(storage[:-2])

    width, height = resolution.split("x")
    df.at[0, "Width"] = int(width)
    df.at[0, "Height"] = int(height)

    _, cpu_mark = get_cpu_name(cpu)
    _, gpu_mark = get_gpu_name(gpu)
    df.at[0, "GPU Mark"] = float(gpu_mark)
    df.at[0, "CPU Mark"] = float(cpu_mark)

    return df

def knn_predict_price(new_data: pd.DataFrame, model_path, scaler_path):
    try:
        knn = joblib.load(open(model_path, "rb"))
        scaler = joblib.load(open(scaler_path, "rb"))

        print(new_data)
        columns_to_scale = ["Weight", "Monitor", "RAM", "Storage Amount", "GPU Mark", "Width", "Height", "CPU Mark"]
        new_data_scaled = pd.DataFrame(scaler.transform(new_data[columns_to_scale]), columns=columns_to_scale)
        new_data[columns_to_scale] = new_data_scaled

        y_pred = knn.predict(new_data)
        print(new_data)
        print(f"Predicted Price: {y_pred[0]}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return float(y_pred[0])


if __name__=="__main__":
    BRAND = "Apple"
    CPU = "Intel Core i7 12700H"
    GPU = "Intel Iris Xe"
    MONITOR = "17.3"
    RESOLUTION = "1920x1080"
    RAM = "16GB"
    STORAGE = "512GB"
    OS = "Windows 11 Home 64-bit"
    WEIGHT = "2.60"

    MODELPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "knn_model_2.pkl"))
    SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "save_scaler.pkl"))

    new_data = transfer_to_df(brand=BRAND,
                          cpu=CPU,
                          gpu=GPU,
                          monitor=MONITOR,
                          resolution=RESOLUTION,
                          ram=RAM,
                          storage=STORAGE,
                          os=OS,
                          weight=WEIGHT)
    
    print("Unscaled Data:")
    print(new_data)

    y_hat = knn_predict_price(new_data=new_data, model_path=MODELPATH, scaler_path=SCALER_PATH)
    print(y_hat)
    print(type(y_hat))