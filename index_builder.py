import os
import json
from collections import defaultdict
from math import log
from bs4 import BeautifulSoup
import hashlib
import pickle  
import sqlite3
import re  
from urllib.parse import urlparse, urlunparse  

from constants import DATA_DIR, PARTIAL_INDEX_DIR, FINAL_INDEX_DIR, ANALYTICS_FILE, PARTIAL_FLUSH_LIMIT, DOC_MAP_FILE, TITLE_MAP_FILE, IDF_FILE
from utils import tokenize, stem_tokens, is_valid, stable_hash_url  # updated import

index_cache = {}

def flush_partial_index(index, flush_id):
    os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True)
    filename = os.path.join(PARTIAL_INDEX_DIR, f"partial_{flush_id}.pkl")  # new line
    with open(filename, 'wb') as f:  # new line
        pickle.dump(index, f)  # new line

def merge_indices(partial_dir):
    final_index = defaultdict(dict)
    for filename in os.listdir(partial_dir):
        if filename.endswith(".pkl"):  # new line
            with open(os.path.join(partial_dir, filename), 'rb') as f:  # new line
                partial = pickle.load(f)  # new line
                for token, postings in partial.items():
                    for doc_id, posting in postings.items():
                        final_index[token][doc_id] = final_index[token].get(doc_id, {"positions": []})
                        final_index[token][doc_id]["positions"].extend(posting.get("positions", []))
    return final_index


def write_analytics(index, doc_count):
    size_kb = sum(os.path.getsize(os.path.join(FINAL_INDEX_DIR, f)) for f in os.listdir(FINAL_INDEX_DIR)) // 1024
    with open(ANALYTICS_FILE, 'w') as f:
        f.write(f"Documents indexed: {doc_count}\n")
        f.write(f"Unique tokens: {len(index)}\n")
        f.write(f"Index size on disk: {size_kb} KB\n")

def write_index_to_sqlite(index):
    conn = sqlite3.connect("final_index.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inverted_index (
            term TEXT PRIMARY KEY,
            postings TEXT
        )
    """)
    for term, postings in index.items():
        cursor.execute(
            "INSERT OR REPLACE INTO inverted_index (term, postings) VALUES (?, ?)",
            (term, json.dumps(postings))
        )
    conn.commit()
    conn.close()

def build_index():
    seen_hashes = set()
    seen_token_sets = []

    temp_index = defaultdict(lambda: defaultdict(list))
    doc_count = 0
    flush_id = 0
    doc_map = {}
    title_map = {}

    if os.path.exists(PARTIAL_INDEX_DIR):
        for f in os.listdir(PARTIAL_INDEX_DIR):
            os.remove(os.path.join(PARTIAL_INDEX_DIR, f))

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                try:
                    page = json.load(f)
                    url = page.get("url", "")
                    if not url or not is_valid(url):
                        continue
                    print(f"Processing document {doc_count + 1}: {url}")
                    norm_url = url
                    doc_id = stable_hash_url(norm_url)
                    if doc_id in doc_map:
                        continue
                    doc_map[doc_id] = norm_url

                    content = page.get("content", "")
                    soup = BeautifulSoup(content, "html.parser")
                    for tag in soup(["header", "footer", "nav", "aside", "script", "style"]):
                        tag.decompose()
                    main = soup.find("main") or soup.find("div", {"id": "main"}) or soup.body
                    text = main.get_text(separator=" ", strip=True) if main else ""

                    content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                    if content_hash in seen_hashes:
                        print(f"Skipped exact duplicate: {url}")
                        continue  # Skip exact duplicate
                    seen_hashes.add(content_hash)

                    shingles = set(text.lower().split())  # basic token split
                    is_near_duplicate = any(len(shingles & prev) / len(shingles | prev) > 0.9 for prev in seen_token_sets)
                    if is_near_duplicate:
                        print(f"Skipped near duplicate: {url}")
                        continue
                    seen_token_sets.append(shingles)

                    tokens = stem_tokens(tokenize(text))

                    for i, token in enumerate(tokens):
                        temp_index[token][doc_id].append(i)
                    doc_count += 1
                except Exception:
                    continue

            if doc_count % PARTIAL_FLUSH_LIMIT == 0:
                flush_partial_index(temp_index, flush_id)
                print(f"Flushed partial index {flush_id} with {doc_count} documents")
                temp_index.clear()
                flush_id += 1

    if temp_index:
        flush_partial_index(temp_index, flush_id)
        print(f"Final flush completed with flush ID {flush_id}")

    with open(DOC_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(doc_map, f)
    with open(TITLE_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(title_map, f)
    print("Saved document and title maps")

    final_index = merge_indices(PARTIAL_INDEX_DIR)
    print("Merged partial indices into final index")
    idf_values = {token: log(doc_count / len(postings)) for token, postings in final_index.items()}
    with open(IDF_FILE, "w", encoding="utf-8") as f:
        json.dump(idf_values, f)
    print("Saved IDF values")

    write_index_to_sqlite(final_index)
    print("Saved final index to SQLite database")
    write_analytics(final_index, doc_count)
    print("Wrote analytics to file")

def load_postings_for_term(term, db_path="final_index.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT postings FROM inverted_index WHERE term=?", (term,))
    row = cursor.fetchone()
    conn.close()
    if row:
        postings = json.loads(row[0])
        return postings, len(postings)
    return {}, 0