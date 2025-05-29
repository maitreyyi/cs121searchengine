import re
import hashlib
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import nltk

from constants import STOPWORDS

nltk.download('punkt')
stemmer = PorterStemmer()

def tokenize(text):
    try:
        soup = BeautifulSoup(text, "html.parser")
        clean_text = soup.get_text(separator=" ", strip=True)
    except Exception:
        clean_text = text
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', clean_text.lower())
    return [token for token in tokens if not token.isdigit() and len(token) > 1 and token not in STOPWORDS]

def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

def process_query_terms(query, remove_stopwords=True):
    tokens = query.lower().split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [stemmer.stem(t) for t in tokens]

def stable_hash_url(url):
    return int(hashlib.md5(url.encode()).hexdigest()[:8], 16)

def normalize_url(url):
    try:
        parsed = urlparse(url)
        normalized = parsed._replace(fragment="", query="").geturl()
        if normalized.endswith("/") and len(normalized) > len(parsed.scheme) + 3 + len(parsed.netloc):
            normalized = normalized.rstrip("/")
        return normalized
    except Exception:
        return url

def is_valid_url(url):
    try:
        p = urlparse(url)
        return all([p.scheme in ("http", "https"), p.netloc])
    except Exception:
        return False