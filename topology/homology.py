# -*- coding: utf-8 -*-
from __future__ import print_function

from spaces import CellComplex

__all__ = [
    'homology'
    ]

    
def homology(K, L=None):
    """Cacul de l'homologie H(K, L) selon [dSLVJ11]."""
    if L is None: L = lambda x: False

    alive = []
    for sigma in (s for k in range(K.dimension(), -1, -1) for s in K.cells(k) if not L(s)):
        update = []
        partner = None
        for (z, dz) in alive:
            if sigma not in dz:
                update.append((z, dz))
            else:
                if partner is None:
                    partner = (z, dz)
                else:
                    y, dy = partner
                    update.append((z ^ y, dz ^ dy))
        if partner is None:
            update.append((set([sigma]), set(tau for tau in K.facets(sigma) if not L(tau))))
        alive = update

    H = {}
    for (z, dz) in alive:
        assert not dz
        cycle = list(z)
        H.setdefault(K.dimension(cycle[0]), []).append(cycle)

    return H


def twist(K, L=None):
    """Calcul de l'homologie H(K, L) selon [CK11].

    K est un dictionnaire qui retourne les faces de chaque simplexe.
    L est une fonction qui retourne vraie si le simplexe est dans le
    sous-complexe L."""
    if L is None: L = lambda x: False

    partner = {}
    D = {sigma: set(tau for tau in bdsigma if not L(tau)) for sigma, bdsigma in K.iteritems() if not L(sigma)}
    DD = {}
    for sigma in sorted(D.iterkeys(), key=lambda x: (len(x), x), reverse=True):
        bdsigma = D.get(sigma, set())
        csigma = {sigma}

        while bdsigma:
            tau = max(bdsigma)
            if tau not in partner:
                # tau est tué par sigma
                D[sigma] = bdsigma
                partner[tau] = sigma
                D.pop(tau, None)              # Le homology -> tau est tué !
                break
            else:
                bdsigma ^= D[partner[tau]]
                csigma ^= DD[partner[tau]]

        DD[sigma] = csigma

    H = defaultdict(list)
    for sigma, tau in D.iteritems():
        if not tau:
            H[len(sigma)-1].append(list(DD[sigma]))
    return H

        

if __name__ == '__main__':
    from algorithm import flag

    # Description du complexe
    
    # 1 -- 2 -- 3 / {1, 3}
    # neighbors = {1: (2,), 2: (1, 3), 3: (2,)}
    # relative = frozenset([1, 3])

    
    # 1 -- 2 -- 3 -- 4 -- 5 + 2--6--7--3 / {1, 4 -- 5}
    # neighbors = {1: (2,), 2: (1, 3, 6), 3: (2, 4, 7), 4: (3, 5),
    #              5: (4,), 6: (2, 7), 7: (2, 3)}
    # relative = frozenset([1, 4, 5])

    
    # 1 -- 2 -- 3 -- 4 -- cycle + 5, 6 connectés à 1, 2, 3 et 4 / {cycle 1,2,3,4}
    neighbors = {1: (2, 4, 5, 6),
                 2: (1, 3, 5, 6),
                 3: (2, 4, 5, 6),
                 4: (1, 3, 5, 6),
                 5: (1, 2, 3, 4, 6),
                 6: (1, 2, 3, 4, 5)}
    relative = frozenset(range(5))

    # Construction du complexe
    K = CellComplex()
    for simplex in flag(neighbors):
        if len(simplex) == 1:
            K.add_cell(list(simplex)[0], relative=all(s in relative for s in simplex))
        elif len(simplex) == 2:
            K.add_cell(simplex, list(simplex), relative=all(s in relative for s in simplex))
        else:
            bord = [frozenset(simplex - set([v])) for v in simplex]
            K.add_cell(simplex, bord, relative=all(s in relative for s in simplex))

    # Calcul de l'homologie
    cycles = homology(K, lambda x: K[x]['relative'])
            
    # Affichage
    print('Complex:')
    for k in range(1+K.dimension()):
        print('  dimension', k)
        for sigma in K.cells(k):
            print('  ', '  ', sigma, K[sigma]['relative'])

    print('\nCycles:')
    for k in cycles.keys():
        print('  dimension', k)
        for cycle in cycles[k]:
            print('  ', '  ', cycle)
