import pandas as pd
import math
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({"feature1" : [1,2,3,4,5],
                  "feature2" : [2,3,5,4,6],
                  "feature3" : [3,2,4,5,6]})

print(df)
print(df.describe())

sc = StandardScaler()
df_scaled = sc.fit_transform(df)

a = math.sqrt(2.5)
print((1-3)/a)
print(1/math.sqrt(2))

print(df.describe())
print(type(df))

new_data = pd.DataFrame({"feature1": [4],
                         "feature2": [6],
                         "feature3": [4]})

new_data_scaled = sc.transform(new_data)

print(df)
print(new_data)
print(df_scaled)
print(new_data_scaled)