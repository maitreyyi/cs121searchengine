import json
import time
from collections import defaultdict
from scoring import full_phrase_in_doc, score_document
from utils import process_query_terms, is_live_url
from constants import DOC_MAP_FILE, TITLE_MAP_FILE, IDF_FILE, DOC_COUNT, HEADING_MAP_FILE
from index_builder import load_postings_for_term
from requests import head

def run_query(query, doc_map, idf_values, title_map, heading_map, test_mode=False):
    terms = process_query_terms(query)
    candidate_docs = []
    postings_dict = {}

    for term in terms:
        postings, df = load_postings_for_term(term)
        if df == 0:
            if test_mode:
                continue
            else:
                print(f"Missing term: {term} in index --- abort")
                return
        postings_dict[term] = postings
        candidate_docs.append(set(postings.keys()))

    if not candidate_docs:
        if test_mode:
            print("No documents matched this query.")
        else:
            print("No documents matched.")
        return

    common_docs = set.union(*candidate_docs)
    if not common_docs:
        if test_mode:
            print("No common documents with any query terms.")
        else:
            print("No common documents with all query terms.")
        return

    docs_to_score = list(common_docs)

    scores = defaultdict(float)
    start_time = time.time()
    phrase_match_count = 0

    for doc_id in docs_to_score:
        url = doc_map.get(str(doc_id), "")

        matched_terms = [term for term in terms if doc_id in postings_dict.get(term, {})]
        if not matched_terms:
            matched_terms = []

        coverage = len(matched_terms) / len(terms)

        is_phrase_match = full_phrase_in_doc(terms, doc_id, postings_dict) if coverage == 1.0 else False

        if is_phrase_match:
            phrase_match_count += 1

        base_score = score_document(
            doc_id, terms, postings_dict, idf_values, title_map, doc_map,
            phrase_boost=(50 if is_phrase_match else 0), require_all_terms=False
        )

        if title_map:
            title = title_map.get(str(doc_id), "").lower()
            if all(term in title for term in terms):
                base_score += 20

        if heading_map:
            headings = heading_map.get(str(doc_id), "").lower()
            if all(term in headings for term in terms):
                base_score += 15

        scores[doc_id] = base_score * coverage

    elapsed = time.time() - start_time
    phrase_ratio = phrase_match_count / len(docs_to_score) if docs_to_score else 0

    if 0.1 < phrase_ratio < 0.9:
        for doc_id in scores:
            scores[doc_id] *= 0.85

    if test_mode:
        print(f"Query: {query}")
        print(f"Query processed in {elapsed * 1000:.2f} ms")
    else:
        print(f"Query processed in {elapsed * 1000:.2f} ms")

    if scores:
        top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        shown = 0
        for doc_id, score in top_docs:
            url = doc_map.get(str(doc_id), "")
            print(f"[DEBUG] Doc {doc_id} score: {score:.2f}")
            print(f"{shown + 1}. {url}")
            shown += 1
            if shown == 5:
                break
    else:
        if test_mode:
            print("No documents matched after scoring.")
        else:
            print("No documents matched.")

    print("-" * 50)

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

    try:
        with open(HEADING_MAP_FILE, "r", encoding="utf-8") as f:
            heading_map = json.load(f)
    except Exception:
        heading_map = {}

    for idx, q in enumerate(test_queries, 1):
        print(f"\n{idx}. Query: {q} ")
        run_query(q, doc_map, idf_values, title_map, heading_map, test_mode=True)

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

    try:
        with open(HEADING_MAP_FILE, "r", encoding="utf-8") as f:
            heading_map = json.load(f)
    except Exception:
        heading_map = {}

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

        run_query(query, doc_map, idf_values, title_map, heading_map)
