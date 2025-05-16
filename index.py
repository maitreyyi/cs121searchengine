'''
Builds inverted index from the given dataset of HTML pages (JSON files)
'''
import os
import json
from bs4 import BeautifulSoup
import nltk
import re
from collections import defaultdict
from nltk.stem import PorterStemmer
import hashlib

nltk.download('punkt')

DATA_DIR = "data"
DOC_MAP_DIR = "doc_maps"
PARTIAL_INDEX_DIR = "partial_indices"
FINAL_INDEX = "index.json"
ANALYTICS_FILE = "analytics.txt"
PARTIAL_FLUSH_LIMIT = 5000

stemmer = PorterStemmer()

def tokenize(text):
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ", strip=True)
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', clean_text.lower())
    return [token for token in tokens if not token.isdigit() and len(token) > 1]

def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

def stable_hash_url(url):
    return int(hashlib.md5(url.encode()).hexdigest()[:8], 16)

def flush_partial_index(index, flush_id):
    os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True)
    filename = os.path.join(PARTIAL_INDEX_DIR, f"partial_{flush_id}.json")

    # Wrap each frequency count in {"tf": value}
    converted_index = {
        token: {doc_id: {"tf": tf} for doc_id, tf in doc_freqs.items()}
        for token, doc_freqs in index.items()
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(converted_index, f)
    print(f"Flushed partial index: {filename}")

def merge_indices(partial_dir):
    if not os.path.isdir(partial_dir):
        print(f"No partial index directory found at {partial_dir}, skipping merge.")
        return defaultdict(dict)

    final_index = defaultdict(dict)

    for filename in os.listdir(partial_dir):
        if filename.endswith(".json"):
            with open(os.path.join(partial_dir, filename), 'r', encoding='utf-8') as f:
                partial = json.load(f)
                for token, postings in partial.items():
                    for doc_id, posting in postings.items():
                        tf = posting.get("tf", 0)
                        if doc_id not in final_index[token]:
                            final_index[token][doc_id] = {"tf": tf}
                        else:
                            final_index[token][doc_id]["tf"] += tf
    return final_index

def write_analytics(index, doc_count):
    index_size_kb = os.path.getsize(FINAL_INDEX) // 1024
    with open(ANALYTICS_FILE, 'w') as f:
        f.write(f"Documents indexed: {doc_count}\n")
        f.write(f"Unique tokens: {len(index)}\n")
        f.write(f"Index size on disk: {index_size_kb} KB\n")

def merge_doc_maps(doc_map_dir):
    merged = {}
    for file in os.listdir(doc_map_dir):
        if file.endswith(".json"):
            with open(os.path.join(doc_map_dir, file), "r", encoding="utf-8") as f:
                merged.update(json.load(f))
    with open("doc_map.json", "w", encoding="utf-8") as f:
        json.dump(merged, f)

def build_index():
    temp_index = defaultdict(lambda: defaultdict(int))
    doc_count = 0
    flush_id = 0
    doc_map = {}

    if os.path.exists(DOC_MAP_DIR):
        for f in os.listdir(DOC_MAP_DIR):
            os.remove(os.path.join(DOC_MAP_DIR, f))


    if os.path.exists(PARTIAL_INDEX_DIR):
        for f in os.listdir(PARTIAL_INDEX_DIR):
            os.remove(os.path.join(PARTIAL_INDEX_DIR, f))

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if not file.endswith(".json"):
                continue
            try:
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    page = json.load(f)
                    
                    try:
                        doc_id = stable_hash_url(page.get("url"))
                        doc_map[doc_id] = page.get("url")
                    except ValueError:
                        continue  # or log and skip problematic URLs

                    content = page.get("content", "")
                    tokens = stem_tokens(tokenize(content))
                    
                    for token in tokens:
                        temp_index[token][doc_id] += 1

                    doc_count += 1
                    print("Doc count: ", doc_count)
            except Exception as e:
                print(f"Error reading {file}: {e}")

            if doc_count % PARTIAL_FLUSH_LIMIT == 0:
                print("Flushing...")
                flush_partial_index(temp_index, flush_id)

                with open(os.path.join(DOC_MAP_DIR, f"doc_map_part_{flush_id}.json"), "w", encoding="utf-8") as f:
                    json.dump(doc_map,f)
                doc_map.clear()
                flush_id += 1
                temp_index.clear()

    if temp_index:
        flush_partial_index(temp_index, flush_id)
        
        os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True)
        with open(os.path.join(DOC_MAP_DIR, f"doc_map_part_{flush_id}.json"), "w", encoding="utf-8") as f:
            json.dump(doc_map, f)
        doc_map.clear()

    print("Merging partial indexes...")
    final_index = merge_indices(PARTIAL_INDEX_DIR)
    write_analytics(final_index, doc_count)
    print(f"Indexing complete. Total documents: {doc_count}")
    merge_doc_maps(DOC_MAP_DIR)

if __name__ == "__main__":
    build_index()
