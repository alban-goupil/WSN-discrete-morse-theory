# -*- coding: utf-8 -*-
from __future__ import print_function

from collections import defaultdict

##------------------------------------------------------------------------------
## Calcul de l'homologie selon [dSLVJ]
def homology(K, L=None):
    """Cacul de l'homologie H(K, L) selon [dSLVJ11].

    K est un dictionnaire qui retourne les faces de chaque simplexe.
    L est une fonction qui retourne vraie si le simplexe est dans le
    sous-complexe L."""
    if L is None: L = lambda x: False
    alive = []
    for sigma in sorted((s for s in K.iterkeys() if not L(s)), key=len, reverse=True):
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
            update.append((set([sigma]), set(tau for tau in K[sigma] if not L(tau))))
        alive = update

    H = defaultdict(list)
    for (z, dz) in alive:
        # assert not dz
        cycle = list(z)
        H[len(cycle[0])-1].append(cycle)

    return H


##------------------------------------------------------------------------------
## Calcul de l'homologie selon [CK11]
def twist(K, L=None):
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
                D.pop(tau, None)              # Le twist -> tau est tué !
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

##------------------------------------------------------------------------------
## Test de l'homologie

def facets(cell):
    """Facets of a simplicial cell; e.g. list(facets((1,2,3))) == [(2,3), (1,3), (1,2)]."""
    for i in range(len(cell)):
        yield cell[:i] + cell[(i+1):]

# Le complexe
K = {f: set(facets(f)) for f in ['ab', 'ad', 'bc', 'bd', 'ce', 'de', 'abd']}

# Le calcul de l'homologie
H = twist(K)
