import os
import sys
import csv

import requests
from bs4 import BeautifulSoup
from random import uniform
from time import sleep
from time import time

from common import access_website_js_render

def get_config_details_from(url: str) -> dict:
    '''
    url: https://www.bhphotovideo.com/c/product/1710341-REG/apple_mbam2mn_15_13_6_macbook_air_m2.html
    specs_url: url + "/specs
    RETURN: Configuration detail of "laptop_link"
        {
            "Operating System": "macOS",
            "Processor": "Apple M2",
            "GPU": "Apple (10-Core)",
            ...
        }
    '''
    specs_url = url + "/specs"
    raw_content = access_website_js_render(url=specs_url)
    if raw_content is None:
        return None
    
    sleep(uniform(3,4))
    
    soup = BeautifulSoup(raw_content, "html.parser")

    config_detail = dict()
    try:
        # Extract name : "HP 16" ZBook Studio G9 Mobile Workstation Wolf Pro Security Edition"
        product_name = ""
        product_name = soup.find("h1", class_="text_TAw0W35QK_").text
        if product_name:
            print("Name :",product_name)
        
        # Extract price: "$1,599.00"
        product_price = ""
        product_price = soup.find("div", class_="price__9gLfjPSjp").text
        if product_price:
            print("Price:",product_price)
        
        # Map product_name to dictionary
        config_detail["Name"] = product_name

        # pair_KqJ3Q3GPKv keySpec_KqJ3Q3GPKv
        config_tags = soup.find_all("tr", class_="pair_KqJ3Q3GPKv")
        print(config_tags)

        for config_tag in config_tags:
            label, value = "", ""
            label = config_tag.find("td", class_="label_KqJ3Q3GPKv").text
            value_tag = config_tag.find("td", class_="value_KqJ3Q3GPKv")
            value = value_tag.find("span").text

            # if value != "":
            #     value = " ".join(value.split())

            if label not in config_detail.keys():
                config_detail[label] = value
        
        config_detail["Price"] = product_price
        print(config_detail)

        return config_detail

    except:
        print("Not found configuration!")

    return config_detail

def write_to_csv(filepath: str, config_detail: dict):
    try:
        header = config_detail.keys()

        with open(filepath, "a", newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=header)

            if file.tell() == 0:
                writer.writeheader()

            writer.writerow(config_detail)

            print(f" > Successful write {config_detail['Name']}")

    except:
        print(f"Error to write: {config_detail['Name']}")

if __name__ == '__main__':
    BASE_PATH = os.path.join(os.path.dirname(__file__), "..")
    PARENT_PATH = os.path.join(BASE_PATH, "..")
    PARENT_PATH = os.path.abspath(os.path.join(PARENT_PATH, ".."))
    RAW_DATA_PATH = os.path.join(PARENT_PATH,"data", "raw")

    FILEPATH = "bhphotovideo_items.txt"
    SCRAPING_DATA_PATH = os.path.join(RAW_DATA_PATH, FILEPATH)

    # Modify 
    START_IDX = 1
    END_IDX = 3

    initial_time = time()
    
    product_links = []
    with open(SCRAPING_DATA_PATH, "r") as f:
        for line in f.readlines():
            product_links.append(line[:-1])
    
    # product_links.__len__() = 1422. So I divide to 9 times for scraping
    # 1 : [0:150]    # 5 : [601:750]
    # 2 : [151:300]  # 6 : [751-900]
    # 3 : [301:450]  # 7 : [901-1050]
    # 4 : [451:600]  # 8 : [1051-1200]
    # 9 : [1201-1421]

    product_links = product_links[START_IDX-1:END_IDX]
    stack = product_links.copy()

    # all_product_configs = []
    while stack.__len__() != 0:
        product_link = stack.pop(0)

        sleep(uniform(1,3))
        config_detail = get_config_details_from(product_link)

        if config_detail is None or config_detail.values().__len__() <= 2:
            stack.append(product_link)
        
        else:
            write_to_csv(filepath="bhphotovideo_v2.csv", config_detail=config_detail)
        
        print(f"Product remain in stack: {stack.__len__()}")

    final_time = time()

    print(f"Execution time: {(final_time - initial_time)/60:.5f} mins / {len(product_links)} products")
    print(f"Last index    : {END_IDX}")