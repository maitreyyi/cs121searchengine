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
import string
from math import log

from urllib.parse import urlparse


nltk.download('punkt')

STOPWORDS = {"a", "an", "the", "of", "on", "in", "for", "and", "to", "with"}

DATA_DIR = "data"
PARTIAL_INDEX_DIR = "partial_indices"
FINAL_INDEX = "index.json"
ANALYTICS_FILE = "analytics.txt"
PARTIAL_FLUSH_LIMIT = 5000

stemmer = PorterStemmer()

def tokenize(text):
    # soup = BeautifulSoup(text, "html.parser")
    # clean_text = soup.get_text(separator=" ", strip=True)
    # try:
    #     soup = BeautifulSoup(text, features="xml")
    #     clean_text = soup.get_text(separator=" ", strip=True)
    # except Exception:
    #     soup = BeautifulSoup(text, "html.parser")
    #     clean_text = soup.get_text(separator=" ", strip=True)
    # tokens = re.findall(r'\b[a-zA-Z0-9]+\b', clean_text.lower())
    # return [token for token in tokens if not token.isdigit() and len(token) > 1]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(text, features="xml")
        clean_text = soup.get_text(separator=" ", strip=True)
    except Exception:
        soup = BeautifulSoup(text, "html.parser")
        clean_text = soup.get_text(separator=" ", strip=True)
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', clean_text.lower())
    return [token for token in tokens if not token.isdigit() and len(token) > 1]


def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

def process_query_terms(query):
    tokens = query.lower().split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return [stemmer.stem(t) for t in tokens]

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

def build_index():
    temp_index = defaultdict(lambda: defaultdict(int))
    doc_count = 0
    flush_id = 0
    doc_map = {}
    
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
                flush_id += 1
                temp_index.clear()

    if temp_index:
        flush_partial_index(temp_index, flush_id)

    with open("doc_map.json", "w", encoding="utf-8") as f:
        json.dump(doc_map, f)

    with open("doc_stats.json", "w", encoding="utf-8") as f:
        json.dump({"doc_count": doc_count}, f)

    print("Merging partial indexes...")
    final_index = merge_indices(PARTIAL_INDEX_DIR)
    split_index_by_prefix(final_index)
    write_analytics(final_index, doc_count)
    print(f"Indexing complete. Total documents: {doc_count}")

def load_postings_for_term(term, index_dir="final_index"):
    prefix = term[0].lower() if term[0].isalpha() else "other"
    filepath = os.path.join(index_dir, f"index_{prefix}.json")

    if not os.path.exists(filepath):
        return {}, 0

    with open(filepath, "r", encoding="utf-8") as f:
        index = json.load(f)

    if term in index:
        postings = {doc_id: posting["tf"] for doc_id, posting in index[term].items()}
        df = len(postings)
        return postings, df
    return {}, 0
    
def is_valid_url(url):
    try:
        p = urlparse(url)
        return all([p.scheme in ("http", "https"), p.netloc])
    except Exception:
        return False

def run_predefined_queries(doc_map, total_docs):
    test_queries = [
        "cristina lopes",
        "machine learning",
        "ACM",
        "master of software engineering"
    ]
    print("\n🔍 Running Predefined Query Tests...\n")
    for q in test_queries:
        terms = process_query_terms(q)

        scores = defaultdict(float)
        for term in terms:
            postings, df = load_postings_for_term(term)
            if df == 0:
                continue
            idf = log(total_docs / df)
            for doc_id, tf in postings.items():
                scores[doc_id] += tf * idf

        print(f"\nQuery: {q}")
        if scores:
            top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (doc_id, score) in enumerate(top_docs, start=1):
                url = doc_map.get(str(doc_id))
                if url and url.startswith("http"):
                    print(f"{i}. {url}")
        else:
            print("No documents matched this query.")
        print("-" * 50)

def search_interface():
    # Load doc_map.json
    try:
        with open("doc_map.json", "r", encoding="utf-8") as f:
            doc_map = json.load(f)
    except Exception as e:
        print(f"Could not load doc_map.json: {e}")
        doc_map = {}

    # Load document stats (total doc count)
    try:
        with open("doc_stats.json", "r", encoding="utf-8") as f:
            stats = json.load(f)
            total_docs = stats.get("doc_count", 1)
    except:
        total_docs = 1

    print("Type 'exit' or 'q' to quit.\n")
    print("Type '/test' to run predefined query evaluation.\n")

    while True:
        query = input("Search: ").strip()
        if query.lower() in {"exit", "q"}:
            print("Exiting search.")
            break

        elif query.lower() == "/test":
            run_predefined_queries(doc_map, total_docs)
            continue

        terms = process_query_terms(query)
        scores = defaultdict(float)
        
        for term in terms:
            postings, df = load_postings_for_term(term)
            if df == 0:
                continue
            idf = log(total_docs / df)
            for doc_id, tf in postings.items():
                scores[doc_id] += tf * idf

        # print(f"\nSearch: {query}")

        if scores:
            # Sort and get up to top 5 results
            top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (doc_id, score) in enumerate(top_docs, start=1):
                url = doc_map.get(str(doc_id))
                if url and url.startswith("http"):
                    print(f"{i}. {url}")
        else:
            print("No documents matched all query terms.")



# Split index by prefix (a-z and 'other'), write index_{prefix}.json files
def split_index_by_prefix(final_index, output_dir="final_index"):
    os.makedirs(output_dir, exist_ok=True)

    split_index = {ch: {} for ch in string.ascii_lowercase}
    split_index["other"] = {}

    for term, postings in final_index.items():
        prefix = term[0].lower()
        if prefix in split_index:
            split_index[prefix][term] = postings
        else:
            split_index["other"][term] = postings

    for prefix, index_chunk in split_index.items():
        path = os.path.join(output_dir, f"index_{prefix}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index_chunk, f)
        print(f"Wrote: {path}")

if __name__ == "__main__":
    # build_index() #run once, only re-run if data changes
    search_interface()