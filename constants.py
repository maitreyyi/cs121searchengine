

import string

# Directory paths
DATA_DIR = "data"
PARTIAL_INDEX_DIR = "partial_indices"
FINAL_INDEX_DIR = "final_index"

# File names
ANALYTICS_FILE = "analytics.txt"
DOC_MAP_FILE = "doc_map.json"
TITLE_MAP_FILE = "title_map.json"
IDF_FILE = "idf.json"
PAGERANK_FILE = "pagerank.json"

# Indexing limits
PARTIAL_FLUSH_LIMIT = 5000
DOC_COUNT = 55393  # Total number of documents

# Stopword list
STOPWORDS = {"a", "an", "the", "of", "on", "in", "for", "and", "to", "with"}

# Valid prefix set for splitting index files
VALID_PREFIXES = set(string.ascii_lowercase)