

import string

# Directory paths
DATA_DIR = "data"
PARTIAL_INDEX_DIR = "partial_indices"

# File names
ANALYTICS_FILE = "analytics.txt"
# Indexing limits
PARTIAL_FLUSH_LIMIT = 5000
DOC_COUNT = 55393  # Total number of documents

# Stopword list
STOPWORDS = {"a", "an", "the", "of", "on", "in", "for", "and", "to", "with"}

# Valid prefix set for splitting index files
VALID_PREFIXES = set(string.ascii_lowercase)