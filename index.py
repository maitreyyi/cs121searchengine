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


FINAL_INDEX = "index.json"
ANALYTICS_FILE = "analytics.txt"
nltk.download('punkt')

STOPWORDS = {"a", "an", "the", "of", "on", "in", "for", "and", "to", "with"}

DATA_DIR = "data"
PARTIAL_INDEX_DIR = "partial_indices"
PARTIAL_FLUSH_LIMIT = 5000
DOC_COUNT = 55393

stemmer = PorterStemmer()

index_cache = {}

def write_analytics(index, doc_count):
    index_size_kb = sum(os.path.getsize(os.path.join("final_index", f)) for f in os.listdir("final_index") if f.endswith(".json")) // 1024
    with open(ANALYTICS_FILE, 'w') as f:
        f.write(f"Documents indexed: {doc_count}\n")
        f.write(f"Unique tokens: {len(index)}\n")
        f.write(f"Index size on disk: {index_size_kb} KB\n")

def tokenize(text):
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

def process_query_terms(query, remove_stopwords=True):
    tokens = query.lower().split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [stemmer.stem(t) for t in tokens]

def stable_hash_url(url):
    return int(hashlib.md5(url.encode()).hexdigest()[:8], 16)

def flush_partial_index(index, flush_id):
    os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True)
    filename = os.path.join(PARTIAL_INDEX_DIR, f"partial_{flush_id}.json")

    # Wrap each posting in {"positions": positions}
    converted_index = {
        token: {doc_id: {"positions": positions} for doc_id, positions in doc_freqs.items()}
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
                        if doc_id not in final_index[token]:
                            final_index[token][doc_id] = {"positions": posting.get("positions", [])}
                        else:
                            final_index[token][doc_id]["positions"].extend(posting.get("positions", []))
    return final_index


def build_index():
    temp_index = defaultdict(lambda: defaultdict(list))
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
            
                    for position, token in enumerate(tokens):
                        temp_index[token][doc_id].append(position)

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

    print("Merging partial indexes...")
    final_index = merge_indices(PARTIAL_INDEX_DIR)

    # Calculate and store IDF values
    idf_values = {
        token: log(doc_count / len(postings)) for token, postings in final_index.items()
    }
    with open("idf.json", "w", encoding="utf-8") as f:
        json.dump(idf_values, f)

    split_index_by_prefix(final_index)
    write_analytics(final_index, doc_count)
    print(f"Indexing complete. Total documents: {doc_count}")

def load_postings_for_term(term, index_dir="final_index"):
    prefix = term[0].lower() if term[0].isalpha() else "other"
    filepath = os.path.join(index_dir, f"index_{prefix}.json")

    if not os.path.exists(filepath):
        return {}, 0

    if filepath in index_cache:
        index = index_cache[filepath]
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            index = json.load(f)
            index_cache[filepath] = index

    if term in index:
        postings = {doc_id: {"positions": posting.get("positions", [])} for doc_id, posting in index[term].items()}
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
    print("\nRunning Predefined Query Tests...\n")
    # Load IDF values
    try:
        with open("idf.json", "r", encoding="utf-8") as f:
            idf_values = json.load(f)
    except Exception:
        idf_values = {}
    for q in test_queries:
        terms = process_query_terms(q, remove_stopwords=False)

        candidate_docs = []
        postings_dict = {}

        for term in terms:
            postings, df = load_postings_for_term(term)
            if df == 0:
                continue
            postings_dict[term] = postings
            doc_ids = set(postings.keys())
            candidate_docs.append(doc_ids)

        if not candidate_docs:
            print(f"\nQuery: {q}")
            print("No documents matched this query.")
            print("-" * 50)
            continue

        common_docs = set.intersection(*candidate_docs)
        scores = defaultdict(float)
        for doc_id in common_docs:
            if phrase_in_doc(terms, doc_id, postings_dict, window_size=4):
                for term in terms:
                    posting = postings_dict[term][doc_id]
                    scores[doc_id] += len(posting["positions"]) * idf_values.get(term, 0)

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

    # Use constant for total document count
    total_docs = DOC_COUNT

    print("Type 'exit' or 'q' to quit.\n")
    print("Type '/test' to run predefined query evaluation.\n")

    # Load IDF values once
    try:
        with open("idf.json", "r", encoding="utf-8") as f:
            idf_values = json.load(f)
    except Exception:
        idf_values = {}

    while True:
        query = input("Search: ").strip()
        if query.lower() in {"exit", "q"}:
            print("Exiting search.")
            break

        elif query.lower() == "/test":
            run_predefined_queries(doc_map, total_docs)
            continue

        terms = process_query_terms(query, remove_stopwords=False)
        candidate_docs = []
        postings_dict = {}

        for term in terms:
            postings, df = load_postings_for_term(term)
            if df == 0:
                continue
            postings_dict[term] = postings
            doc_ids = set(postings.keys())
            candidate_docs.append(doc_ids)

        if not candidate_docs:
            print("No documents matched all query terms.")
            continue

        common_docs = set.intersection(*candidate_docs)
        scores = defaultdict(float)
        for doc_id in common_docs:
            if phrase_in_doc(terms, doc_id, postings_dict, window_size=4):
                for term in terms:
                    posting = postings_dict[term][doc_id]
                    scores[doc_id] += len(posting["positions"]) * idf_values.get(term, 0)

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

# Phrase-in-document helper for phrase search (proximity-based)
def phrase_in_doc(terms, doc_id, index, window_size=4):
    try:
        positions_lists = [index[term][str(doc_id)]["positions"] for term in terms]
    except KeyError:
        return False

    # Flatten and sort all positions
    all_positions = sorted(pos for plist in positions_lists for pos in plist)

    # Check for any window of size len(terms) where max - min <= window_size
    for i in range(len(all_positions) - len(terms) + 1):
        window = all_positions[i + len(terms) - 1] - all_positions[i]
        if window <= window_size:
            return True
    return False

if __name__ == "__main__":
    # build_index() #run once, only re-run if data changes
    search_interface()
