import json
import time
from collections import defaultdict
from scoring import phrase_in_doc, full_phrase_in_doc, score_document
from utils import process_query_terms
from constants import DOC_MAP_FILE, TITLE_MAP_FILE, IDF_FILE, DOC_COUNT
from index_builder import load_postings_for_term


def run_predefined_queries(doc_map, total_docs):
    test_queries = [
        "cristina lopes",
        "machine learning",
        "ACM",
        "master of software engineering"
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

    for q in test_queries:
        print(f"\nQuery: {q}")
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
            continue

        common_docs = set.intersection(*candidate_docs)
        phrase_docs = [doc_id for doc_id in common_docs if full_phrase_in_doc(terms, doc_id, postings_dict)]

        scores = defaultdict(float)
        for doc_id in phrase_docs:
            scores[doc_id] = score_document(doc_id, terms, postings_dict, idf_values, title_map, doc_map)

        top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (doc_id, _) in enumerate(top_docs, 1):
            print(f"{i}. {doc_map.get(str(doc_id), '')}")
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

    print("Type 'exit' or 'q' to quit.")
    print("Type '/test' to run predefined queries.\n")

    while True:
        query = input("Search: ").strip()
        if query.lower() in {"exit", "q"}:
            break
        if query.lower() == "/test":
            run_predefined_queries(doc_map, DOC_COUNT)
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
        phrase_docs = [doc_id for doc_id in common_docs if full_phrase_in_doc(terms, doc_id, postings_dict)]

        scores = defaultdict(float)
        for doc_id in phrase_docs:
            scores[doc_id] = score_document(doc_id, terms, postings_dict, idf_values, title_map, doc_map)

        print(f"Query processed in {(time.time() - start) * 1000:.2f} ms")
        if scores:
            top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (doc_id, _) in enumerate(top_docs, 1):
                print(f"{i}. {doc_map.get(str(doc_id), '')}")
        else:
            print("No documents matched.")