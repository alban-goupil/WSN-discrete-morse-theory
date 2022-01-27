# -*- coding: utf-8 -*-
from __future__ import print_function

import math
import random

from itertools import chain

from topology.spaces import CellComplex
from topology.algorithm import flag

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


#random.seed(1234)                       # Couverture (1 2-cycle sur le relatif)
random.seed(12345)                      # 1 1-cycle
#random.seed(1236)                       # 2 1-cycle2




##------------------------------------------------------------------------------
## Paramètres
Nf = 10                                        # Nombre de sommets de la bordure
Nc = 40                                        # Nombre de sommets intérieurs
r = 0.65                                       # Rayon de connexion
N = Nf + Nc                                    # Nombre total de sommets


##------------------------------------------------------------------------------
## Calcul de l'homologie selon [dSLVJ11]
def homology(K, L=None):
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


##------------------------------------------------------------------------------
## Calcul de l'homologie selon la théorie de Morse discrète [For98, For02, CGN13]

## Flow I + dV + Vd
def flow(K, V, c):
    chain = set(c)                      # I
    for sigma in c:                     # + Vd
        chain.symmetric_difference_update(V.get(tau) for tau in K.facets(sigma))
        if sigma in V:                  # + dV
            chain.symmetric_difference_update(K.facets(V.get(sigma)))
    chain.discard(None)
    return chain


## Itération du flot pour atteindre le point fixe
def flow_invariant(K, V, c):
    prev = set()
    chain = set(c)
    while prev <> chain:
        prev, chain = chain, flow(K, V, chain)
    return chain


## Construction du complexe de Morse
def morse(K, V, A=None):
    # Critical cells if needed
    if A is None:
        A = set(K.cells())
        A.difference_update(V.iterkeys())
        A.difference_update(V.itervalues())

    # Morse Complex
    Mp = CellComplex()
    for sigma in sorted(A, key=lambda x: K.dimension(x)):
        # Recherche l'invariant associé à la cellule critique
        chain = flow_invariant(K, V, [sigma])
        dchain = flow_invariant(K, V, list(K.facets(sigma)))
    
        # Ajout à Mp
        Mp.add_cell(sigma, list(dchain & A), dim=K.dimension(sigma))

    return Mp


def random_field(K, key=None):
    if key is None:
        clef = lambda x: K.dimension(x)
    else:
        clef = lambda x: (K.dimension(x), key(x))
    
    V = dict()
    P = set(K.cells())
    A = set()

    while P:
        # Recherche d'une paire libre
        pairs = []
        for sigma in P:
            cobd = P.intersection(K.cofacets(sigma))
            if len(cobd) == 1:
                pairs.append((sigma, cobd.pop()))

        # Utilisation de la meilleure paire
        if pairs:
            sigma, tau = min(pairs, key=lambda st: clef(st[1]))
            V[sigma] = tau
            P.remove(sigma)
            P.remove(tau)
        else:
            # Création d'une cellule critique pour V
            sigma = max(P, key=clef)
            A.add(sigma)
            P.remove(sigma)

    return V, A


##------------------------------------------------------------------------------
## Positionnement des sommets et relation de voisinage et complexe

# La bordure
phi = 2 * math.pi / Nf
locations = [(math.cos(i * phi), math.sin(i * phi)) for i in range(Nf)]

# Ajout des autres points
while len(locations) < N:
    x = random.uniform(-1.0, 1.0)
    y = random.uniform(-1.0, 1.0)
    if x*x + y*y < 1.0:
        locations.append((x, y))


# Voisinage
neighbors = {s: set() for s in range(N)}

for s in range(Nf):                       # Fence
    neighbors[s].add((s+1) % Nf)
    neighbors[s].add((Nf + s-1) % Nf)

for s in range(N):                        # Tous
    xs, ys = locations[s]
    for t in range(s+1, N):
        xt, yt = locations[t]
        if math.hypot(xs - xt, ys - yt) < r:
            neighbors[s].add(t)
            neighbors[t].add(s)


# Complexe
K = CellComplex()
for simplex in flag(neighbors, 4):
    simplex = tuple(sorted(simplex))
    faces = [tuple(sorted(set(simplex) - {v})) for v in simplex]
    K.add_cell(simplex, faces if len(simplex) > 1 else [],
               onFence=all(s < Nf for s in simplex))
    # Calcul de la taille 
    dim = len(simplex)-1
    if dim == 1:
        zs = [complex(*locations[s]) for s in simplex]
        K[simplex]['size'] = abs(zs[0] - zs[1])
    elif dim == 2:
        zs = [complex(*locations[s]) for s in simplex]
        # formule d'Héron numériquement plus stable (Kahan)
        c, b, a = sorted([abs(zs[0] - zs[1]), abs(zs[0] - zs[2]), abs(zs[1] - zs[2])])
        K[simplex]['size'] = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    else: # dim == 0 or dim > 3:
        K[simplex]['size'] = 0.0


##------------------------------------------------------------------------------
## Calcul de H1(K) et de H2(K|Fence)
H = homology(K)
Hr = homology(K, lambda x: K[x]['onFence'])


##------------------------------------------------------------------------------
## Calcul de H1(K)
V, A = random_field(K) # , key=lambda x: -K[x]['size'])
Mp = morse(K, V, A)
HMp = homology(Mp)
Hp = {d: [flow_invariant(K, V, c) for c in Hd] for d, Hd in HMp.iteritems()}

Cp = dict()
for c in A:
    Cp.setdefault(len(c)-1, []).append(c)

##------------------------------------------------------------------------------
# Affichage texte
print('Betti')
for k, v in H.iteritems():
    print('  ', k, len(v))

print('Betti par Vector field')
for k, v in Cp.iteritems():
    print('  ', k, len(v))

print('Betti par Morse')
for k, v in Hp.iteritems():
    print('  ', k, len(v))
    
print('Relative Betti')
for k, v in Hr.iteritems():
    print('  ', k, len(v))


##------------------------------------------------------------------------------
# Affichage graphique
a = plt.axes()
a.axis([-1.0, 1.0, -1.0, 1.0])

if K.dimension() > 0:
    for sigma in K.cells(1):
        xs, ys = zip(*[locations[s] for (s,) in K.vertices(sigma)])
        p = mlines.Line2D(xs, ys, color='green', lw=1)
        a.add_artist(p)

if K.dimension() > 1:
    for sigma in K.cells(2):
        xys = [locations[s] for (s,) in K.vertices(sigma)]
        p = mpatches.Polygon(xys, fill=True, facecolor='blue', alpha=0.2)
        a.add_artist(p)
        
for cycle in H.get(1, []):
    for e in cycle:
        xs, ys = zip(*[locations[s] for (s,) in K.vertices(e)])
        p = mlines.Line2D(xs, ys, color='red', lw=4, alpha=0.4)
        a.add_artist(p)

for cycle in Hp.get(1, []):
    for e in cycle:
        xs, ys = zip(*[locations[s] for (s,) in K.vertices(e)])
        p = mlines.Line2D(xs, ys, color='yellow', lw=4, alpha=0.4)
        a.add_artist(p)

awoken = set()
if 2 in Hr:
    if len(Hr[2]) != 1:
        print("Erreur de conception H_2(K|F) n'est pas monogène")
    awoken = set(chain.from_iterable(Hr[2][0]))
        
for i, (x, y) in enumerate(locations):
    a.text(x, y, "%2d" % i, fontsize=12, color='red' if i in awoken else 'black')

    
plt.show(block=False)
