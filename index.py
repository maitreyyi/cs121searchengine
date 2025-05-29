"""
Main entry point for building the index and running the search interface.
"""

from index_builder import build_index
from search import search_interface

if __name__ == "__main__":
    # build_index() # Uncomment to build index
    search_interface()
