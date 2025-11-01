# <script async src = "https://cse.google.com/cse.js?cx=10ba722503d954a97" >
# </script >
# <div class = "gcse-search" > </div >
"AIzaSyBQPQnVK-BnNzGss0JSsESxPr_OoPhYLoI"


import requests
from dotenv import load_dotenv
import os
load_dotenv()


def query_google(board_game):
    API_KEY = os.getenv("GOOGLE_API_KEY")
    SEARCH_ENGINE_ID = os.getenv("GOOGLE_CX_KEY")
    query = f"{board_game} rules"

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": 5,
    }

    r = requests.get(url, params=params)
    results = r.json().get("items", [])

    for i, item in enumerate(results, 1):
        print(f"{i}. {item['title']}")
        print(item['link'])
        print(item['snippet'], "\n")
