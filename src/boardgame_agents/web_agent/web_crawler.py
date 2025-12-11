import requests
from dotenv import load_dotenv
import os
from pathlib import Path
from pdfminer.high_level import extract_text
from transformers import AutoTokenizer


load_dotenv()


def query_google(search_term, postfix="board game rules pdf"):
    API_KEY = os.getenv("GOOGLE_API_KEY")
    SEARCH_ENGINE_ID = os.getenv("GOOGLE_CX_KEY")
    query = f"{search_term} {postfix}"

    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": 5,
    }

    try:
        r = requests.get(base_url, params=params)
    except Exception as e:
        raise (f"{e}")

    results = r.json().get("items", [])

    for i, item in enumerate(results, 1):
        url = item["link"]
        if url.split(".")[-1] == "pdf":
            return save_pdf(url, search_term)


def save_pdf(url, search_term, save_dir="pdfs"):
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if url.lower().endswith(".pdf"):
        filename = save_dir.joinpath(f"{search_term}.pdf")
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            with open(filename, "wb") as f:
                f.write(r.content)
            return extract_text_from_pdf(filename)
            # print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Failed: {url} ({e})")


def extract_text_from_pdf(pdf_path, max_tokens=5000):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    text = extract_text(pdf_path)

    tokens = tokenizer.encode(text)

    tokens = tokens[:max_tokens]

    return tokenizer.decode(tokens)
