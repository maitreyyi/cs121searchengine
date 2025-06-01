import json
import time
from collections import defaultdict
from scoring import phrase_in_doc, full_phrase_in_doc, score_document
from utils import process_query_terms
from constants import DOC_MAP_FILE, TITLE_MAP_FILE, IDF_FILE, DOC_COUNT
from index_builder import load_postings_for_term
from requests import head

def is_url_alive(url):
    try:
        res = head(url, timeout=3, allow_redirects=True)
        return res.status_code < 400
    except:
        return False

def run_predefined_queries(doc_map, total_docs, test):
    test_queries = []
    if test == 0:
        test_queries = [
            "cristina lopes",
            "machine learning",
            "ACM",
            "master of software engineering"
        ]
    elif test == 1:
        test_queries = [
            "cristina lopes",
            "master of software engineering",
            "machine learning",
            "computer science degree",
            "informatics uc irvine",
            "software engineering curriculum",
            "how to apply for mswe",
            "school of information and computer sciences",
            "ics faculty list",
            "acm icpc competition",
            "data science tracks",
            "academic integrity policy",
            "website accessibility standards",
            "cs course prerequisites",
            "uci parking pass",
            "cafeteria menu",
            "uc irvine housing info",
            "undergraduate vs graduate",
            "student research paper format",
            "staff office hours",
        ]

    try:
        with open(IDF_FILE, "r", encoding="utf-8") as f:
            idf_values = json.load(f)
    except Exception:
        idf_values = {}

    try:
        with open(TITLE_MAP_FILE, "r", encoding="utf-8") as f:
            title_map = json.load(f)
    except Exception:
        title_map = {}

    for idx, q in enumerate(test_queries, 1):
        print(f"\n{idx}. Query: {q} ")
        start_time = time.time()
        terms = process_query_terms(q)
        candidate_docs = []
        postings_dict = {}

        for term in terms:
            postings, df = load_postings_for_term(term)
            if df == 0:
                continue
            postings_dict[term] = postings
            candidate_docs.append(set(postings.keys()))

        if not candidate_docs:
            print("No documents matched this query.")
            print("-" * 50)
            continue

        common_docs = set.intersection(*candidate_docs)
        if not common_docs:
            print("No common documents with all query terms.")
            print("-" * 50)
            continue

        docs_to_score = list(common_docs)

        scores = defaultdict(float)
        for doc_id in docs_to_score:
            is_phrase_match = full_phrase_in_doc(terms, doc_id, postings_dict)
            scores[doc_id] = score_document(
                doc_id, terms, postings_dict, idf_values, title_map, doc_map,
                phrase_boost=(1000 if is_phrase_match else 0)
            )

        top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        elapsed = time.time() - start_time
        print(f"Query processed in {elapsed * 1000:.2f} ms")

        if top_docs:
            shown = 0
            for doc_id, _ in top_docs:
                url = doc_map.get(str(doc_id), "")
                if not is_url_alive(url):
                    continue
                shown += 1
                print(f"{shown}. {url}")
                if shown == 5:
                    break

        else:
            print("No documents matched after scoring.")
        print("-" * 50)


def search_interface():
    try:
        with open(DOC_MAP_FILE, "r", encoding="utf-8") as f:
            doc_map = json.load(f)
    except Exception:
        doc_map = {}

    try:
        with open(TITLE_MAP_FILE, "r", encoding="utf-8") as f:
            title_map = json.load(f)
    except Exception:
        title_map = {}

    try:
        with open(IDF_FILE, "r", encoding="utf-8") as f:
            idf_values = json.load(f)
    except Exception:
        idf_values = {}

    print("\nSearch Engine Project")      
    print("What do you want to look for today?\n")

    print("Type 'm2' to run A3:M2 predefined queries.")
    print("Type 'm3' to run A3:M3 predefined queries.\n")
    print("Type 'exit' or 'q' to quit.")

    while True:
        query = input("Search: ").strip()
        if query.lower() in {"exit", "q"}:
            break
        if query.lower() == "m2":
            run_predefined_queries(doc_map, DOC_COUNT, 0)
            continue
        if query.lower() == "m3":
            run_predefined_queries(doc_map, DOC_COUNT, 1)
            continue

        start = time.time()
        terms = process_query_terms(query)
        candidate_docs = []
        postings_dict = {}

        for term in terms:
            postings, df = load_postings_for_term(term)
            if df == 0:
                print(f"Missing term: {term} in index --- abort")
                candidate_docs = []
                break
            postings_dict[term] = postings
            candidate_docs.append(set(postings.keys()))

        if not candidate_docs:
            print("No documents matched.")
            continue

        common_docs = set.intersection(*candidate_docs)
        if not common_docs:
            print("No common documents with all query terms.")
            continue

        docs_to_score = list(common_docs)

        scores = defaultdict(float)
        for doc_id in docs_to_score:
            is_phrase_match = full_phrase_in_doc(terms, doc_id, postings_dict)
            scores[doc_id] = score_document(
                doc_id, terms, postings_dict, idf_values, title_map, doc_map,
                phrase_boost=(1000 if is_phrase_match else 0)
            )

        print(f"Query processed in {(time.time() - start) * 1000:.2f} ms")
        if scores:
            top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            shown = 0
            for doc_id, _ in top_docs:
                url = doc_map.get(str(doc_id), "")
                if not is_url_alive(url):
                    continue
                shown += 1
                print(f"{shown}. {url}")
                if shown == 5:
                    break

        else:
            print("No documents matched.")
