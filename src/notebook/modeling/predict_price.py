import pandas as pd
from sklearn.preprocessing import StandardScaler
from map_cpu_gpu import *
import joblib

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


def predict_price(new_data: pd.DataFrame, loaded_model="") -> float:
    SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "x_train_scaler.pkl"))

    scaler = joblib.load(open(SCALER_PATH, "rb"))

    columns_to_scale = ["Weight", "Monitor", "RAM", "Storage Amount", "GPU Mark", "Width", "Height", "CPU Mark"]
    
    new_data_scaled = scaler.transform(new_data, columns_to_scale)
    # y_pred = loaded_model.predict(new_data_scaled)
    print(new_data_scaled)
    # return y_pred


if __name__=="__main__":
    BRAND = "Apple"
    CPU = "Intel Core i7 12700U"
    GPU = "Intel Iris Xe"
    MONITOR = "15.6"
    RESOLUTION = "1920x1080"
    RAM = "8GB"
    STORAGE = "256GB"
    OS = "macOS"
    WEIGHT = "1.78"

    SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "knn_default_scaler.pkl"))
    MODELPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "knn_default_model.pkl"))

    new_data = transfer_to_df(brand=BRAND,
                          cpu=CPU,
                          gpu=GPU,
                          monitor=MONITOR,
                          resolution=RESOLUTION,
                          ram=RAM,
                          storage=STORAGE,
                          os=OS,
                          weight=WEIGHT)
    
    print(new_data)


    try:
        scaler = joblib.load(open(SCALER_PATH, "rb"))
        KNN_MODEL = joblib.load(open(MODELPATH, "rb"))

        columns_to_scale = ["Weight", "Monitor", "RAM", "Storage Amount", "GPU Mark", "Width", "Height", "CPU Mark"]
        new_data_scaled = pd.DataFrame(scaler.transform(new_data[columns_to_scale]), columns=columns_to_scale)
        new_data[columns_to_scale] = new_data_scaled

        print(new_data)
        y_pred = KNN_MODEL.predict(new_data)
        print(new_data)
        print(y_pred)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
