# -*- coding: utf-8 -*-

from itertools import chain
from collections import namedtuple, defaultdict
from _utils import UnionFind

Interval = namedtuple('Interval', "start end cycle")

class PairCells:
    """Implementation of the Pair-Cells algorithm of [Zom09]."""
    def __init__(self, K, ctime):
        cells = sorted(K.cells(), key=lambda c: (ctime(c), K.dimension(c)))
        time = {}
        index = {}
        partner = {}
        cascade = {}
        bdcascade = {}
        self._pairs = []

        # PairCell algorithm
        for i, sigma in enumerate(cells):
            time[sigma] = ctime(sigma)
            index[sigma] = i

            csigma = set([sigma])
            bdsigma = set(K.facets(sigma))

            while bdsigma:
                tau = max(bdsigma, key=index.get)
                if tau not in partner:
                    # tau is destroyed by sigma
                    self._pairs.append((tau, sigma))
                    partner[sigma] = tau
                    partner[tau] = sigma
                    break
                else:
                    csigma ^= cascade[partner[tau]]
                    bdsigma ^= bdcascade[partner[tau]]

            cascade[sigma] = csigma
            bdcascade[sigma] = bdsigma

        # Pairs for undestroyed cells
        self._pairs.extend((s, None) for s in cells if s not in partner)

        # Save cascade and partner
        self._cascade = cascade
        self._partner = partner


    def intervals(self, dim=None):
        # Intervals computation on demand
        if self._intervals is None:
            self._intervals = defaultdict(list)
            for sigma, tau in self._pairs:
                d = K.dimension(sigma)
                self._intervals[d].append(Interval(time[sigma],
                                                   time[tau] if tau is not None else None,
                                                   cascade[sigma]))
        if dim is None:
            return chain(*self._intervals.values())
        return iter(self._intervals[dim])

    
    def pairs(self):
        return iter(self._pairs)

    def cascade(self, sigma):
        return self._cascade[sigma]

    def partner(self, sigma):
        return self._partner.get(sigma)

    
class PCoh:
    """Implementation of the persistence algorithm pCoh given in [dSMVJ11] and
    also explained in [dSMVJ11b]."""
    def __init__(self, K, ctime):
        cells = sorted(K.cells(), key=lambda c: (ctime(c), K.dimension(c)))
        index = {}
        self._pairs = []
        
        Z = {}

        # pCoh algorithm
        for i, sigma in enumerate(cells):
            index[sigma] = i

            bd_sigma = set(K.facets(sigma))

            upsilon = None
            indices = []
            for tau in Z:
                # Calcul < d Z[tau], sigma > = < Z[tau], d sigma >
                if len (bd_sigma.intersection(Z[tau])) % 2 != 0:
                    if upsilon is None or index[tau] > index[upsilon]:
                        upsilon = tau
                    indices.append(tau)
            
            if indices:
                for tau in indices:
                    if tau is not upsilon:
                        Z[tau] ^= Z[upsilon]
                del Z[upsilon]
                self._pairs.append((upsilon, sigma))
            else:
                Z[sigma] = set([sigma])

        self._pairs.extend((s, None) for s in Z)


    def intervals(self, dim=None):
        # Intervals computation on demand
        if self._intervals is None:
            self._intervals = defaultdict(list)
            for sigma, tau in self._pairs:
                d = K.dimension(sigma)
                self._intervals[d].append(Interval(time[sigma],
                                                   time[tau] if tau is not None else None,
                                                   cascade[sigma]))
        if dim is None:
            return chain(*self._intervals.values())
        return iter(self._intervals[dim])

    def pairs(self):
        return iter(self._pairs)


class PairCellsSurfaceWithoutBoundary:
    """Implementation of Homological Persistence for surface without boundary given in [BLW12]."""
    def __init__(self, K, ctime):
        assert K.dimension() <= 2
        self._intervals = {}
        self._pairs = []

        cells = sorted(K.cells(), key=lambda c: (ctime(c), K.dimension(c)))
        index = {}

        # passage cellule -> temps/index
        for i, sigma in enumerate(cells):
            index[sigma] = i

        # Find (0, 1) persistence
        representatives = {}
        components = UnionFind()
        for sigma in cells:
            dim_sigma = K.dimension(sigma)
            if dim_sigma == 0:
                representatives[sigma] = sigma
                components.add(sigma)
            elif dim_sigma == 1:
                a, b = K.facets(sigma)
                ac = components[a]
                bc = components[b]

                # Merge the components if needed
                if ac != bc:
                    nc = components.union(ac, bc)
                    yac = representatives.pop(ac, ac)
                    ybc = representatives.pop(bc, bc)
                    if index[ybc] < index[yac]:
                        yac, ybc = ybc, yac # assert yac < ybc < sigma
                    representatives[nc] = yac
                    self._pairs.append((ybc, sigma))

        for c in representatives.values():
            self._pairs.append((c, None))

        # find (1, 2) persistence using duality
        representatives = {}
        components = UnionFind()
        for sigma in reversed(cells):
            dim_sigma = K.dimension(sigma)
            if dim_sigma == 2:
                representatives[sigma] = sigma
                components.add(sigma)
            elif dim_sigma == 1:
                a, b = K.cofacets(sigma)
                ac = components[a]
                bc = components[b]

                # Merge the components if needed
                if ac != bc:
                    nc = components.union(ac, bc)
                    yac = representatives.pop(ac, ac)
                    ybc = representatives.pop(bc, bc)
                    if index[ybc] > index[yac]:
                        yac, ybc = ybc, yac # assert sigma < ybc < yac
                    representatives[nc] = yac
                    self._pairs.append((sigma, ybc))

        for c in representatives.values():
            self._pairs.append((c, None))

        # Intervals computation
        self._intervals = {}
        for sigma, tau in self._pairs:
            d = K.dimension(sigma)
            self._intervals.setdefault(d,[]).append(Interval(ctime(sigma),
                                                             None if tau is None else ctime(tau),
                                                             None))

    def intervals(self, dim=None):
        if dim is None:
            return chain(*self._intervals.values())
        return iter(self._intervals[dim])

    def pairs(self):
        return iter(self._pairs)
