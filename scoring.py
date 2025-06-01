from math import log
from collections import defaultdict

idf_cache = {}

def get_idf(term, total_docs, index):
    if term in idf_cache:
        return idf_cache[term]
    df = len(index.get(term, {}))
    idf = log((total_docs + 1) / (df + 1))
    idf_cache[term] = idf
    return idf


def score_document(doc_id, terms, postings_dict, idf_values, title_map=None, doc_map=None, phrase_boost=1000):
    # ✅ Ensure the document contains all query terms in the body
    if any(doc_id not in postings_dict.get(term, {}) for term in terms):
        return 0.0

    score = 0.0
    doc_len = sum(
        len(postings_dict[t][doc_id]["positions"])
        for t in terms if doc_id in postings_dict[t]
    )

    for term in terms:
        if doc_id in postings_dict[term]:
            freq = len(postings_dict[term][doc_id]["positions"])
            tfidf = (freq / doc_len) * idf_values.get(term, 0) if doc_len > 0 else 0
            score += tfidf

    # ✅ Only apply URL and title boosts *after* ensuring main content has all terms
    if doc_map:
        url = doc_map.get(str(doc_id), "")
        url_lower = url.lower()
        for term in terms:
            if term in url_lower:
                score += 15
            if term in url:
                score += 10
        score -= url.count('/')

    if title_map:
        title = title_map.get(str(doc_id), "").lower()
        for term in terms:
            if term in title:
                score += 20

    # ✅ Phrase boost is only applied if explicitly passed in (e.g., 1000 for phrase matches, 0 otherwise)
    score += phrase_boost

    return score


def phrase_in_doc(terms, doc_id, index, window_size=4):
    try:
        positions_lists = [index[term][str(doc_id)]["positions"] for term in terms]
    except KeyError:
        return False
    if any(len(plist) > 2000 for plist in positions_lists):
        return False

    all_positions = sorted(pos for plist in positions_lists for pos in plist)
    for i in range(len(all_positions) - len(terms) + 1):
        window = all_positions[i + len(terms) - 1] - all_positions[i]
        if window <= window_size:
            return True
    return False


def full_phrase_in_doc(terms, doc_id, postings_dict):
    try:
        doc_id = str(doc_id)
        positions_lists = [postings_dict[term][doc_id]["positions"] for term in terms]
    except KeyError:
        return False

    first_positions = positions_lists[0]
    for pos in first_positions:
        if all((pos + i) in positions_lists[i] for i in range(1, len(terms))):
            return True
    return False
