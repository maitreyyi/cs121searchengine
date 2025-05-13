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

nltk.download('punkt')

DATA_DIR = "data"
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
    #porter stemming
    return [stemmer.stem(token) for token in tokens]

#write in-memory index to disk periodically instead of holding the entire inverted index in memory until the end
def flush_partial_index(index, flush_id):
    os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True)
    filename = os.path.join(PARTIAL_INDEX_DIR, f"partial_{flush_id}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(index, f)
    print(f"Flushed partial index: {filename}")

#merging the flushes
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
                    for doc_id, tf in postings.items():
                        final_index[token][doc_id] = final_index[token].get(doc_id, 0) + tf
    return final_index

def write_analytics(index, doc_count):
    index_size_kb = os.path.getsize(FINAL_INDEX) // 1024
    with open(ANALYTICS_FILE, 'w') as f:
        f.write(f"Documents indexed: {doc_count}\n")
        f.write(f"Unique tokens: {len(index)}\n")
        f.write(f"Index size on disk: {index_size_kb} KB\n")

def build_index():
    temp_index = defaultdict(lambda: defaultdict(int))
    doc_count = 0
    flush_id = 0

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if not file.endswith(".json"): #invalid file
                continue
            try:
                with open(os.path.join(root, file), "r", encoding = "utf-8") as f:
                    page = json.load(f)
                    doc_id = page.get("url")
                    content = page.get("content","")
                    tokens = stem_tokens(tokenize(content)) #stemming tokens
                    for token in tokens:
                        print("Doc id: ", doc_id, " token: ", token)
                        temp_index[token][doc_id] +=1
                    doc_count +=1
                    print("Doc count: ", doc_count)
            except Exception as e:
                print(f"Error reading {file}: {e}")
            
            if doc_count % PARTIAL_FLUSH_LIMIT == 0:
                print("Flushing...")
                flush_partial_index(temp_index, flush_id)
                flush_id +=1
                temp_index.clear()
    if temp_index:
        flush_partial_index(temp_index, flush_id)

    print("Merging partial indexes...")
    final_index = merge_indices(PARTIAL_INDEX_DIR)

    with open(FINAL_INDEX, 'w', encoding='utf-8') as f:
        json.dump(final_index, f)

    write_analytics(final_index, doc_count)
    print(f"Indexing complete. Total documents: {doc_count}")


if __name__ == "__main__":
    build_index()