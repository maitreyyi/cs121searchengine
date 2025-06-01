import re
import hashlib
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import nltk
import requests

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
    
# is_valid function implementation
def is_valid(url):
    valid_domains = [
        "ics.uci.edu", "cs.uci.edu", "informatics.uci.edu",
        "stat.uci.edu", "today.uci.edu/department/information_computer_sciences"
    ]

    trap_keywords = [
        "/calendar", "/event", "?action=login", "timeline?", "/history", "rev=", "version=", "/diff?version=", "?share=", "/?afg", "/img_", ".ppsx", "/git", "sort=", "orderby=",
        "/print/", "/export/", "/preview/", "/feed/", "sandbox", "staging", "test=", "/archive/", "/archives/", "/version/", "/versions/",
        "mailto:", "share=", "/backup/", "/mirror/", "admin=", "user=", "auth=", "captcha", "trackback", "?sessionid=", "?token=",
        "releases/", "src/", "source/", ".svn/", "/build/", "/dist/", "/static/", "/tmp/", "/text-base/", "/props/", "/prop-base/", "/format", "/all-wcprops",
        ".sql", "/attachment"
    ]

    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        if not any(parsed.netloc.endswith(domain) for domain in valid_domains):
            return False

        low_value_dirs = ["precision", "test", "demo", "features", "output", "logs"]
        if url.lower().endswith(".txt"):
            if any(f"/{dir}/" in url.lower() for dir in low_value_dirs):
                return False

        for keyword in trap_keywords:
            if keyword == ".sql" and url.lower().endswith(".sql"):
                return False
            if keyword == "/attachment" and "/attachment" in url:
                return False
            if keyword in url:
                # Special handling for "releases/"
                if keyword == "releases/" and not re.search(r"/releases/.+\.(html?|txt)$", url):
                    return False
                if keyword != "releases/":
                    return False
        return not re.match(
            r".*\.(css|js|bmp|gif|jpe?g|ico"
            r"|png|tiff?|mid|mp2|mp3|mp4"
            r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
            r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
            r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
            r"|epub|dll|cnf|tgz|sha1"
            r"|thmx|mso|arff|rtf|jar|csv"
            r"|rm|smil|wmv|swf|wma|zip|rar|gz|img|ppsx)$",
            parsed.path.lower()
        )
    except TypeError:
        return False

def is_live_url(url):
    """Returns True if the URL responds with a status_code < 400."""
    try:
        res = requests.head(url, timeout=3, allow_redirects=True)
        return res.status_code < 400
    except Exception:
        return False