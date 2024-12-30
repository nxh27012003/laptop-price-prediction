
import pandas as pd
import pickle
import os
# Now you can import your module
from map_cpu_gpu import *

def predict(t):
    current_dir =os.path.dirname(('\\').join(os.path.abspath(__file__).split('\\')))#file save in src/app
    
    file_path = current_dir+'\encoded_brand_dict.pkl' #relative path to file
    file_path2 = current_dir + '\encoded_OS_dict.pkl' #relative path to file
    
    with open(file_path, 'rb') as file:
        encoded_brand_dict = pickle.load(file)
    with open(file_path2, 'rb') as file:
        encoded_OS_dict = pickle.load(file)
    data_str=t
    values = data_str.split(',')

    columns = ['Brand', 'CPU Name', 'CPU Mark', 'GPU Name', 'GPU Mark', 'Monitor', 'Width', 'Height',
            'RAM', 'Storage Amount', 'OS', 'Weight'] 

    data = pd.DataFrame([values], columns=columns)
    data.drop(columns=['CPU Name','GPU Name'],axis=1,inplace=True)
    data['Brand'] = data['Brand'].str.lower()
    data['Encoded_Brand'] = data['Brand'].apply(lambda x: encoded_brand_dict.get(x) if x in encoded_brand_dict else None)
    data.drop(columns=['Brand'],axis=1,inplace=True)
    data['Encoded_OS'] = data['OS'].apply(lambda x: encoded_OS_dict.get(x) if x in encoded_OS_dict else 100)
    data.drop(columns=['OS'],axis=1,inplace=True)
    data['Resolution'] = int(data['Width'])*int(data['Height'])
    data.drop(columns=['Width','Height'],axis=1,inplace=True)
    
    file_path3 = current_dir+'\saved_model_random_forest.pkl' #relative path to file
    
    with open(file_path3, 'rb') as file:
        loaded_model = pickle.load(file)

    features = ['CPU Mark', 'GPU Mark', 'Monitor', 'RAM', 'Storage Amount','Encoded_Brand','Encoded_OS', 'Resolution']

    X_new_input = data[features]
    predictions_new_input = loaded_model.predict(X_new_input)

    return round(predictions_new_input[0],2)


def rf_predict_price(brand: str='Apple',cpu: str='Intel Core i7-11800H',gpu: str='Intel Iris Xe',monitor: str='15.6',resolution: str='1920x1080',ram: str='8GB',storage: str='256GB',os: str='mac0S',weight: str='1.78'):
    _, cpu_mark = get_cpu_name(cpu)
    _, gpu_mark = get_gpu_name(gpu)
    width,height = resolution.split('x')
    ram = ram.replace('GB','')
    storage = storage.replace('GB','')
    text = brand+","+cpu+","+str(cpu_mark)+","+gpu+","+str(gpu_mark)+","+monitor+","+width+","+height+","+ram+","+storage+","+os+","+weight
    print(text)
    Y_pred = predict(text)
    return Y_pred

if __name__=='__main__':
    print(rf_predict_price())
