import os
import json
import string
from collections import defaultdict
from math import log
from bs4 import BeautifulSoup
import hashlib

from constants import DATA_DIR, PARTIAL_INDEX_DIR, FINAL_INDEX_DIR, ANALYTICS_FILE, PARTIAL_FLUSH_LIMIT, DOC_MAP_FILE, TITLE_MAP_FILE, IDF_FILE
from utils import tokenize, stem_tokens, normalize_url, is_valid_url, stable_hash_url

index_cache = {}

def flush_partial_index(index, flush_id):
    os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True)
    filename = os.path.join(PARTIAL_INDEX_DIR, f"partial_{flush_id}.json")
    converted_index = {
        token: {doc_id: {"positions": positions} for doc_id, positions in doc_freqs.items()}
        for token, doc_freqs in index.items()
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(converted_index, f)

def merge_indices(partial_dir):
    final_index = defaultdict(dict)
    for filename in os.listdir(partial_dir):
        if filename.endswith(".json"):
            with open(os.path.join(partial_dir, filename), 'r', encoding='utf-8') as f:
                partial = json.load(f)
                for token, postings in partial.items():
                    for doc_id, posting in postings.items():
                        final_index[token][doc_id] = final_index[token].get(doc_id, {"positions": []})
                        final_index[token][doc_id]["positions"].extend(posting.get("positions", []))
    return final_index

def split_index_by_prefix(final_index):
    os.makedirs(FINAL_INDEX_DIR, exist_ok=True)
    split_index = {ch: {} for ch in string.ascii_lowercase}
    split_index["other"] = {}
    for term, postings in final_index.items():
        prefix = term[0].lower()
        if prefix in split_index:
            split_index[prefix][term] = postings
        else:
            split_index["other"][term] = postings
    for prefix, index_chunk in split_index.items():
        path = os.path.join(FINAL_INDEX_DIR, f"index_{prefix}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index_chunk, f)

def write_analytics(index, doc_count):
    size_kb = sum(os.path.getsize(os.path.join(FINAL_INDEX_DIR, f)) for f in os.listdir(FINAL_INDEX_DIR)) // 1024
    with open(ANALYTICS_FILE, 'w') as f:
        f.write(f"Documents indexed: {doc_count}\n")
        f.write(f"Unique tokens: {len(index)}\n")
        f.write(f"Index size on disk: {size_kb} KB\n")

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
                    if not url or not is_valid_url(url):
                        continue
                    norm_url = normalize_url(url)
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
                        continue  # Skip exact duplicate
                    seen_hashes.add(content_hash)

                    shingles = set(text.lower().split())  # basic token split
                    is_near_duplicate = any(len(shingles & prev) / len(shingles | prev) > 0.9 for prev in seen_token_sets)
                    if is_near_duplicate:
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
                temp_index.clear()
                flush_id += 1

    if temp_index:
        flush_partial_index(temp_index, flush_id)

    with open(DOC_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(doc_map, f)
    with open(TITLE_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(title_map, f)

    final_index = merge_indices(PARTIAL_INDEX_DIR)
    idf_values = {token: log(doc_count / len(postings)) for token, postings in final_index.items()}
    with open(IDF_FILE, "w", encoding="utf-8") as f:
        json.dump(idf_values, f)

    split_index_by_prefix(final_index)
    write_analytics(final_index, doc_count)

def load_postings_for_term(term, index_dir=FINAL_INDEX_DIR):
    prefix = term[0].lower() if term[0].isalpha() else "other"
    path = os.path.join(index_dir, f"index_{prefix}.json")
    if not os.path.exists(path):
        return {}, 0
    if path in index_cache:
        index = index_cache[path]
    else:
        with open(path, "r", encoding="utf-8") as f:
            index = json.load(f)
            index_cache[path] = index
    if term in index:
        postings = {doc_id: {"positions": posting.get("positions", [])} for doc_id, posting in index[term].items()}
        return postings, len(postings)
    return {}, 0