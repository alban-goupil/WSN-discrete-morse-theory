# -*- coding: utf-8 -*-

from collections import deque

__all__ = [
    'max_cliques',
    'flag'
    ]

def max_cliques(N, k=None):
    """Bron-Kerbosch algorithm generates all max-cliques of the graph described
    by the neighborhood relation N. Neighbors of vertices v is given by
    the set N[v]. If k is given, it limits the search to clique of
    maximal size k.
    """
    def bk(R, P, X):
        if len(R) == k:
            yield frozenset(R)
        else:
            if not any((P, X)):
                yield frozenset(R)
            while P:
                v = P.pop()
                for r in bk(R | set([v]), P & set(N[v]), X & set(N[v])):
                    yield r
                X.add(v)

    return bk(set(), set(N.keys()), set())

def flag(N, k=None):
    q = deque([(frozenset(), set(N.keys()))])

    while q:
        R, P = q.popleft()
        if len(R) == k: continue

        while P:
            v = P.pop()
            r = R | set([v])
            yield r
            q.append((r, P & set(N[v])))
    

if __name__ == '__main__':
    neighbors = {1: (2, 5),
                 2: (1, 3, 5),
                 3: (2, 4),
                 4: (3, 5, 6),
                 5: (1, 2, 4),
                 6: (4,)}


    print("All cells")
    for cell in flag(neighbors):
        print(' ', cell)

    print("\nAll cliques")
    for clique in max_cliques(neighbors):
        print(' ', clique)
