# -*- coding: utf-8 -*-

from itertools import chain  
from collections import defaultdict, deque
from ._utils import PQueue, UnionFind

__all__ = [
    'DiscreteVectorField',
    'RWS11',
    'BLW12'
    ]

class DiscreteVectorField:
    def __init__(self, K):
        self._K = K
        self._vectors = dict()
        self._inverse = dict()
        self._criticals = defaultdict(set)
        for sigma in K.cells():
            self._criticals[K.dimension(sigma)].add(sigma)

    def complex(self):
        return self._K

    def vectors(self):
        """Return the view of the gradient vectors."""
        return self._vectors.items()

    def criticals(self, dim=None):
        """Provides a view of all criticals cells."""
        if dim is None:
            return chain(*self._criticals.values())
        return iter(self._criticals[dim])

    def add_vector(self, a, b):
        # assert a not in self._vectors and b not in self._inverse
        # assert b not in self._vectors and a not in self._inverse
        if self._K.dimension(a) < self._K.dimension(b):
            self._vectors[a] = b
            self._inverse[b] = a
        else:
            self._vectors[b] = a
            self._inverse[a] = b
        if self.is_critical(a): 
            self._criticals[self._K.dimension(a)].remove(a)
        if self.is_critical(b): 
            self._criticals[self._K.dimension(b)].remove(b)

    def remove_vector(self, a):
        if a in self._vectors:
            b = self._vectors.pop(a)
            del self._inverse[b]
            self._criticals[self._K.dimension(a)].add(a)
            self._criticals[self._K.dimension(b)].add(b)
        elif a in self._inverse:
            b = self._inverse.pop(a)
            del self._vectors[b]
            self._criticals[self._K.dimension(a)].add(a)
            self._criticals[self._K.dimension(b)].add(b)

    def remove_cell(self, sigma):
        self.remove_vector(sigma)
        self._criticals[self._K.dimension(sigma)].remove(sigma)

    def is_critical(self, sigma):
        return sigma in self._criticals[self._K.dimension(sigma)]

    def forward(self, sigma):
        if sigma in self._vectors:
            yield self._vectors[sigma]
        for tau in self._K.facets(sigma):
            if tau not in self._vectors or self._vectors[tau] != sigma:
                yield tau

    def backward(self, sigma):
        if sigma in self._inverse:
            yield self._inverse[sigma]
        for tau in self._K.cofacets(sigma):
            if tau not in self._inverse or self._inverse[tau] != sigma:
                yield tau

    def bfs(self, start):
        queue = deque([(start, None)])
        while queue:
            alpha, alphaprev = queue.popleft()
            yield alpha, alphaprev
            for beta in self._K.facets(alpha):
                yield beta, alpha
                if beta in self._vectors:
                    gamma = self._vectors[beta]
                    if gamma != alpha:
                        queue.append((gamma, beta))

    def cancel(self, creator, destroyer):
        # find *the* path
        queue = deque([destroyer])
        parents = {}
        while queue:
            alpha = queue.popleft()
            fs = self._K.facets(alpha)
            if creator in fs:
                parents[creator] = alpha
                break
            for beta in fs:
                if beta not in parents:
                    parents[beta] = alpha
                    if beta in self._vectors:
                        gamma = self._vectors[beta]
                        parents[gamma] = beta
                        queue.append(gamma)
                    
        # Compute the path
        path = [creator]
        sigma = creator
        while sigma != destroyer:
            sigma = parents[sigma]
            path.append(sigma)
            
        # Cancel the path
        for a, b in zip(path[0::2], path[1::2]):
            self.add_vector(a, b)


class _LowerStar:
    def __init__(self, k, x, key):
        self._K = k
        self._cells = set()
        self._pairs = dict()

        mx = max(k.vertices(x), key=key)
        for c in k.star([x]):
            if mx == max(k.vertices(c), key=key): # Add f to lowerstar
                self._cells.add(c)

    def cells(self, dim=None):
        if dim is None:
            return iter(self._cells)
        return iter(sigma for sigma in self._cells
                    if self._K.dimension(sigma) == dim)

    def match(self, a, b=None):
        if b:
            self._pairs[a] = b
            self._pairs[b] = a
        else:
            self._pairs[a] = a

    def is_matchable(self, c):
        return c in self and self._num_unpaired_faces(c) == 1

    def is_alone(self, c):
        return c in self and self._num_unpaired_faces(c) == 0

    def partner(self, c):
        return next(f for f in self._K.facets(c)
                    if f in self and f not in self._pairs)

    def _num_unpaired_faces(self, c):
        return sum(1 for f in self._K.facets(c)
                   if f in self and f not in self._pairs)

    def __len__(self):
        return len(self._cells)

    def __contains__(self, item):
        return item in self._cells


def RWS11(K, key):
    V = DiscreteVectorField(K)
    ckey = lambda c: sorted(map(key, K.vertices(c)))

    # Vector Field Construction
    for x in K.cells(0):
        l = _LowerStar(K, x, key)
        if len(l) == 1:
            l.match(x)
            continue

        pqzero = PQueue(l.cells(1), ckey)
        alpha = pqzero.pop()
        l.match(x, alpha)
        V.add_vector(x, alpha)
        pqone = PQueue((a for a in K.cofacets(alpha) if l.is_matchable(a)),
                       ckey)

        while pqone or pqzero:
            while pqone:
                alpha = pqone.pop()
                if l.is_alone(alpha):
                    pqzero.push(alpha)
                else:
                    beta = l.partner(alpha)
                    l.match(alpha, beta)
                    V.add_vector(alpha, beta)
                    pqzero.remove(beta)

                    for gamma in chain(K.cofacets(alpha), K.cofacets(beta)):
                        if l.is_matchable(gamma):
                            pqone.push(gamma)

            if pqzero:
                alpha = pqzero.pop()
                l.match(alpha)
                for gamma in K.cofacets(alpha):
                    if l.is_matchable(gamma):
                        pqone.push(gamma)
    return V


def BLW12(K, pairs, delta, time):
    edges = {}
    representatives = {}
    components = UnionFind()

    # Build the persistence sub-tree
    for sigma, tau in pairs:
        if tau is not None and time(tau) - time(sigma) <= delta:
            if K.dimension(tau) == 1:
                # For (0, 1) persistence
                a, b = K.facets(tau)

                # Merge the components
                ac = components[a]
                bc = components[b]
                nc = components.union(ac, bc)

                # Keep the book of the representatives element of each component
                # For (0, 1) the representative is the youngest element
                yac = representatives.pop(ac, ac)
                ybc = representatives.pop(bc, bc)
                representatives[nc] = min(yac, ybc, key=time)

                # Add the edge into the tree
                edges.setdefault(a, []).append((b, tau))
                edges.setdefault(b, []).append((a, tau))

            elif K.dimension(tau) == 2 and len(K.cofacets(sigma)) == 2:
                # For (1, 2) persistence
                a, b = K.cofacets(sigma)

                # Merge the components
                ac = components[a]
                bc = components[b]
                nc = components.union(ac, bc)

                # Keep the book of the representatives element of each component
                # For (1, 2) the representative is the eldest element
                yac = representatives.pop(ac, ac)
                ybc = representatives.pop(bc, bc)
                representatives[nc] = max(yac, ybc, key=time)

                # Add the edge into the tree
                edges.setdefault(a, []).append((b, sigma))
                edges.setdefault(b, []).append((a, sigma))

    # Build the Discrete Vector Field
    V = DiscreteVectorField(K)

    stack = [(c, None) for c in representatives.values()]
    while stack:
        a, p = stack.pop()
        for b, e in edges.get(a, []):
            if b != p:
                stack.append((b, a))
                V.add_vector(b, e)

    return V
