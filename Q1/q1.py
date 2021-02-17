import pandas as pd
import random
import urllib3
from bs4 import BeautifulSoup as soup
import re

path = "distance.xlsx"


class FileErr(Exception):
    pass


def openXLS(path):
    try:
        keywords =  pd.read_excel(path)
        return keywords
    except Exception as e:
        raise FileErr() from e

def getPage(url, eaxtraHeaders):

    usrAgnt = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/{random.randint(0, 1000)}.{random.randint(0, 50)} (KHTML, like Gecko) Chrome/{random.randint(10, 99)}.{random.randint(10, 99)}.{random.randint(1000, 9999)}.{random.randint(100, 999)} Safari/{random.randint(100, 999)}.{random.randint(10, 99)}"

    if eaxtraHeaders == None:
        headers = { 'User-Agent' : usrAgnt }
    else:
        headers = ({ 'User-Agent' : usrAgnt }).update(eaxtraHeaders)

    http = urllib3.PoolManager()
    response = http.request('GET', url, headers = headers)
    data = soup(response.data, "html.parser")

    #print(extractBBCLinks(data))
    return data

def saveTextFile(name, contents, subfolderName = None):
    import os.path

    PROJECT_ROOT_DIR = "."
    CHAPTER_ID = "BBC_pages"

    if subfolderName == None:
        save_path = os.path.join(PROJECT_ROOT_DIR, "data", CHAPTER_ID)
    else:
        save_path = os.path.join(PROJECT_ROOT_DIR, "data", CHAPTER_ID, subfolderName)

    os.makedirs(save_path, exist_ok=True)


    completeName = os.path.join(save_path, name + ".html")

    with open(completeName, "w", encoding='utf-8') as file:
        try:
            file.write(contents)
        except:
            print("file write issue")


def extractBBCLinks(data):
    links = []
    containers = data.findAll("div", {"class": "ssrcss-v4rel9-PromoContent e1f5wbog0"})
    for c in containers:
        url_section = c.findAll("div", {"class": "ssrcss-1uw1j0b-PromoHeadline e1f5wbog2"})
        if c.a["href"].split("/")[3] == "news":
            links.append(c.a["href"])

    return list(set(links))


def searchPages():
    """
    goes though the XLM file and searches each item on bbc news and saves the resulting page in a text file
    :return:
    """
    keywords = openXLS(path)['Keywords']
    testURL = "https://www.bbc.co.uk/search?q=durham"

    for word in list(keywords):
        seach_header = {'q': f"{word}"}
        content  = getPage(testURL, seach_header)
        saveTextFile(word, str(content), "base_searches")



def BBCsearch(word):
    """
    produces all links to articles from a certain search on BBC
    :param word:
    :return:
    """
    links = []
    for p in range(30):
        url = f"https://www.bbc.co.uk/search?q={word}&page={p}"
        links.extend(extractBBCLinks(getPage(url, None)))

    return links




if __name__ == '__main__':
    keywords = openXLS(path)['Keywords']
    testURL = "https://www.bbc.co.uk/search"


    #testing
    #keywords = ["Durham"]

    for word in list(keywords):
        links = []

        first_word = 0
        while len(links) < 100 and first_word <= len(word):
            s_parts = word.split(" ")[first_word:]
            searchword = ""
            for part in s_parts:
                searchword += (part + " ")
            searchword = searchword[:-1]


            links.extend(BBCsearch(searchword))
            links = list(set(links))
            first_word += 1

        i = 0
        for link in links:
            i += 1
            content  = getPage(link, None)

            saveTextFile(f"{word}_page_{i}", str(content), f"{word}_pages")


