from collections import defaultdict

def compute_pagerank(outlinks_map, damping=0.85, max_iter=50, tol=1e-6):
    N = len(outlinks_map)
    if N == 0:
        return {}

    pagerank = {doc: 1 / N for doc in outlinks_map}
    inlinks_map = defaultdict(set)

    for src, outlinks in outlinks_map.items():
        for dst in outlinks:
            inlinks_map[dst].add(src)

    for _ in range(max_iter):
        new_rank = {}
        for doc in pagerank:
            rank_sum = sum(
                pagerank[src] / len(outlinks_map[src])
                for src in inlinks_map[doc]
                if len(outlinks_map[src]) > 0
            )
            new_rank[doc] = (1 - damping) / N + damping * rank_sum

        # Check for convergence
        if max(abs(new_rank[doc] - pagerank[doc]) for doc in pagerank) < tol:
            break
        pagerank = new_rank

    return pagerank