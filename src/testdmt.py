# -*- coding: utf-8 -*-
from random import random, seed

seed(1237)

def homology(faces):
    """Cacul de l'homologie H(K) selon [dSLVJ11]."""
    alive = []
    for sigma in sorted(faces.iterkeys(), key=len, reverse=True):
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
            update.append((set([sigma]), faces[sigma]))
        alive = update

    H = {}
    for (z, dz) in alive:
        assert not dz
        cycle = list(z)
        H.setdefault(len(cycle[0])-1, []).append(cycle)

    return H


def boundary(faces, chain):
    dchain = set()
    for sigma in chain: dchain ^= faces[sigma]
    return dchain


## Flow I + dV + Vd
def flowstep(faces, V, c):
    chain = set(c)                      # I
    chain.symmetric_difference_update(V[tau] for tau in boundary(faces, chain) if len(V[tau]) > len(tau))  # + Vd
    chain.symmetric_difference_update(boundary(faces, (V[tau] for tau in chain if len(V[tau]) > len(tau)))) # + dV
    return chain
    

##------------------------------------------------------------------------------
## Test sur l'exemple de [For98]

# Complexe
# faces = {c: set() for c in 'abcde'}
# faces.update((e, set(e)) for e in ['ab', 'ac', 'bc', 'bd', 'cd', 'de'])
# faces['abc'] = set(['ab', 'ac', 'bc'])

from topology.algorithm import flag
neigh = {'a': 'befgo', 'b': 'acghi', 'c': 'bdijk', 'd': 'ceklm', 'e': 'admno',
         'f': 'ago', 'g': 'abfh', 'h': 'bgi', 'i': 'bchj', 'j': 'cik',
         'k': 'cdjl', 'l': 'dkm', 'm': 'ednl', 'n': 'emo', 'o': 'aefn'}

faces = dict()
for simplex in flag(neigh):
    bd = set(''.join(sorted(simplex - {s})) for s in simplex)
    faces[''.join(sorted(simplex))] = bd if len(simplex) > 1 else set()


# Champ aléatoire de vecteurs discrets (selon [BL13b])
potentials = set(faces.iterkeys())
vectors = {}
criticals = set()

while potentials:
    # Recherche d'une paire libre
    for sigma in potentials:
        cobd = potentials.intersection(tau for (tau, dtau) in faces.iteritems() if sigma in dtau)
        if len(cobd) == 1: 
            tau = cobd.pop()
            potentials.remove(sigma)
            potentials.remove(tau)
            vectors[sigma] = tau
            vectors[tau] = sigma
            break
    else:
        # Création d'une cellule critique par manque de paire libre
        sigma = max(potentials, key=lambda x: (len(x), random()))
        vectors[sigma] = sigma
        criticals.add(sigma)
        potentials.remove(sigma)
    

# Calcul du complexe de Morse
morse = {}
for sigma in sorted(criticals, key=len):
    # Recherche l'invariant associé à la cellule critique
    prev = set()
    chain = set([sigma])
    while prev <> chain:
        prev, chain = chain, flowstep(faces, vectors, chain)
    
    # Ajout à Mp
    morse[sigma] = set(boundary(faces, chain) & criticals)

# Classement des cellules critiques par dimension
Cp = dict()
for c in criticals:
    Cp.setdefault(len(c)-1, []).append(c)

# Calcul de l'homologie
Hp = homology(morse)


##------------------------------------------------------------------------------
## Affichage
print 'Cellules critiques', criticals
for k, v in Cp.iteritems():
    print '  ', k, len(v)

print 'Betti'
for k, v in Hp.iteritems():
    print '  ', k, len(v)
