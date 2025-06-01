# validate_urls.py
import json
import requests
from utils import normalize_url

INPUT = "doc_map.json"
OUTPUT = "doc_map_cleaned.json"

def is_url_alive(url):
    try:
        res = requests.head(url, timeout=3, allow_redirects=True)
        return res.status_code < 400
    except:
        return False

with open(INPUT, "r", encoding="utf-8") as f:
    doc_map = json.load(f)

cleaned_map = {}
for doc_id, url in doc_map.items():
    normalized = normalize_url(url)
    if is_url_alive(normalized):
        cleaned_map[doc_id] = normalized
    else:
        print(f"[✗] {normalized} is dead or unreachable.")

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(cleaned_map, f, indent=2)

print(f"\n✅ Cleaned doc_map written to {OUTPUT}")
