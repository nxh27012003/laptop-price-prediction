
import requests
from bs4 import BeautifulSoup
from time import sleep
from random import uniform


def access_website(url: str) -> list:
    API_KEY = "ceff3229258c2096cfb481055b4f0d80b86887cd"
    PARAMS = {
        "url": url,
        "apikey": API_KEY,
    }
    print(f"Accessing {url}...")
    response = requests.get("https://api.zenrows.com/v1/", params=PARAMS)
    if response.status_code == 200:
        print(f"\033[92mSuccessful access\033[00m {url}")
    else:
        print(f"\033[31mAccess deny\033[00m {url}")
        return None
    response.close()
    wait_ = uniform(3,5)
    print(f"Waiting {wait_}s...")
    sleep(wait_)
    return response.content


def access_website_js_render(url: str) -> list:
    API_KEY = "e73e258b4e826462018494c03e029bb203e126af"
    PARAMS = {
        "url": url,
        "apikey": API_KEY,
        "js_render": "true", 
    }
    
    print(f"Accessing {url}...")
    response = requests.get("https://api.zenrows.com/v1/", params=PARAMS)
    wait_ = uniform(3,5)
    print(f"Waiting {wait_}s...")
    sleep(wait_)
    if response.status_code == 200:
        print(f"\033[92mSuccessful access\033[00m {url}")
    
    else:
        print(f"\033[31mAccess deny\033[00m {url}")
        return None
    response.close()
    return response.content


def get_subpages_from(base_url: str, numbers: list, page_num="/pn/_") -> list:

    subpages = [None]
    for pgn in numbers:
        subpage = base_url + page_num.replace("_", str(pgn))
        subpages.append(subpage)

    return subpages


def write_txt(filepath: str, product_links: list):
    '''
    filepath: in the same folder of the current file
    links: list of product_links
    links=[
        https://www.bhphotovideo.com/c/product/1793833-REG/apple_mbp14m345sl_14_macbook_pro_m3.html,
        https://www.bhphotovideo.com/c/product/1793822-REG/apple_mbp14m340sb_14_macbook_pro_m3.html,
        ...
    ]
    write each product in a line of file .txt
    '''
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            for link in product_links:
                f.write(f"{link}\n")
        
        print(f"Successful saving to {filepath}")
    
    except:
        print(f"Error with {filepath}")