import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.drop(df.columns[0], axis=1, inplace=True)

    # Filter Brand
    df["Brand"] = df["Brand"].apply(lambda x: x.lower())
    brands = df["Brand"].value_counts()
    df = df[df["Brand"].isin(brands[brands>7].index.tolist())]
    df["Brand"] = df["Brand"].replace({
        "gigabyte": "other",
        "razer": "other",
        "rokc": "other",
        "best notebooks": "other"
    })
    df[df["Brand"]=="apple"]["OS"].value_counts()
    df.loc[df["Brand"] == "apple", "OS"] = "MacOS"
    
    # Filter CPU Brand
    df["CPU Brand"] = df["CPU Name"].str.split().apply(lambda x: x[0].lower())
    id2 = df["CPU Brand"].value_counts()
    df = df[df["CPU Brand"].isin(id2[id2> 22].index)]

    # Filter GPU Brand
    df["GPU Brand"] = df["GPU Name"].str.split().apply(lambda x: x[0].lower())
    id3 = df["GPU Brand"].value_counts()
    df = df[df["GPU Brand"].isin(id3[id3>5].index)]
    df["GPU Brand"].replace("256mb", "radeon", inplace=True)
    df["GPU Brand"].replace("t550", "t500", inplace=True)
    df["GPU Brand"].replace("geforce3", "geforce", inplace=True)
    df.drop("CPU Name", axis=1, inplace=True)
    df.drop("GPU Name", axis=1, inplace=True)

    # Filter Monitor
    df = df[df["Monitor"]>10.5]
    df = df[df["Monitor"]<20]

    # Filter Width & Height
    df = df[df["Width"]>100]
    df = df[df["Width"]>1300]

    # Filter RAM
    df = df[df["RAM"]>3]

    # Filter Storage
    df = df[df["Storage Amount"]>=32]
    df = df[df["Storage Amount"]<16000]

    # Filter Weight
    df = df[df["Weight"]<7.1]
    df = df[df["Weight"]>0.3]

    # Filter Price
    df = df[df["Price"]>150]
    df = df[df["Price"]<6500]

    # Filter OS
    df["OS"] = df["OS"].replace({
        "ChromeOS": "Chrome OS",
        "No OS": "Windows 10",
        "Windows 8.1": "Windows 8"
    })
    id4 = df["OS"].value_counts()
    df = df[df["OS"].isin(id4[id4>5].index)]

    return df

def encoding_data(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        df = pd.get_dummies(df, columns=[col])
    return df

if __name__ == "__main__":
    print()