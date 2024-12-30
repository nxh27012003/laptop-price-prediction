import os
import sys

from common import *

import requests
from bs4 import BeautifulSoup
from time import sleep
from random import uniform

def get_product_links_from(url: str):
    raw_content = access_website(url=url)
    if raw_content is None:
        return None
    soup = BeautifulSoup(raw_content, "html.parser")
    product_links = list()
    try:
        a_tags = soup.find_all("a", class_="title_UCJ1nUFwhh")
        for a_tag in a_tags:
            product_link = "https://www.bhphotovideo.com" + a_tag.get("href")
            product_links.append(product_link)
    except:
        print("Not found!")
    
    return product_links


# ===== SAMPLE USE (DONOT CHANGE) =====
if __name__ == "__main__":
    BASE_URL = "https://www.bhphotovideo.com/c/buy/laptops/ci/18818"

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RAW_FOLDER = os.path.join(BASE_DIR, "data", "raw")
    SAVE_PATH = "bhphotovideo.txt"
    
    # SCRAPING FROM PAGE 1 -> PAGE 51
    subpages = get_subpages_from(base_url=BASE_URL, numbers=range(1, 52), page_num="/pn/_")
    stack = subpages.copy()

    while stack.__len__() != 1:
        subpage = stack.pop(0)
        links = get_product_links_from(subpage)

        if links == None:
            stack.append(subpage)

        else:
            write_txt(filepath=SAVE_PATH, product_links=links)