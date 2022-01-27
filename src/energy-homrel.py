# -*- coding: utf-8 -*-
from __future__ import print_function

from collections import defaultdict, deque

import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D


##------------------------------------------------------------------------------
## Paramètres

# Dimensionnement
rayon = 0.3                    # Rayon de connexion
Nb = None                      # # capteurs sur le bord, selon rayon
Nt = None                      # # capteurs au total selon le placement

# Échantillonnage 
rmin = 0.1                     # Distance minimum pour le placement des capteurs
k = 30                         # Nombre de points de test

# Perte énergétique
credit = 100.0                 # Énergie initiale selon loi exponentielle

seed = 1234                    # Graine du GNA

##------------------------------------------------------------------------------
## Fonctions pour l'homologie et l'espace des données

def flag(neigh, k=None):
    """Simplices dimension maximum k du flag donné par la relation de voisinage neigh.
    """
    q = deque([(frozenset(), set(neigh.keys()))])

    while q:
        R, P = q.popleft()

        if len(R) == k: continue
        while P:
            v = P.pop()
            r = R | {v}
            yield tuple(sorted(r))
            q.append((r, P & set(neigh[v])))


def facets(cell):
    """Facets of a simplicial cell; e.g. list(facets((1,2,3))) == [(2,3), (1,3), (1,2)]."""
    for i in range(len(cell)):
        yield cell[:i] + cell[(i+1):]


def homology(K, L=None):
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


##------------------------------------------------------------------------------
## Placement des capteurs dans le disque de rayon 1 selon "Fast Poisson Disk
## Sampling in Arbitrary Dimensions" par Robert Bridson [Bri07]
## La positions finales des capteurs sont dans 'locations'

# Setup
ncells = int(np.ceil(2.0 * np.sqrt(2) / rmin))
grid = np.empty((ncells, ncells), dtype=np.complex)
grid[:] = np.infty

rnd.seed(seed)

# Points initiaux
Nb = np.ceil(2*np.pi/rayon)
locs = list(np.exp(2j * np.pi * np.arange(Nb) / Nb))
for p in locs:
    grid[int((p.real + 1.1) / 2.2 * ncells), int((p.imag + 1.1) / 2.2 * ncells)] = p

# Points suivants
actives = list(range(len(locs)))
while actives:
    i = rnd.randint(len(actives))
    actives[i], actives[-1] = actives[-1], actives[i]
    p = locs[actives[-1]]
    for q in p + rmin*np.sqrt(1.0+3.0*rnd.sample(k))*np.exp(2j*np.pi*rnd.sample(k)):
        if abs(q) > 1.0: continue
        qi, qj = int((q.real + 1.1) / 2.2 * ncells), int((q.imag + 1.1) / 2.2 * ncells)
        if (np.abs(grid[max(qi-2,0):min(qi+3, ncells), max(qj-2,0):min(qj+3, ncells)] - q) > rmin).all():
            actives.append(len(locs))
            locs.append(q)
            grid[qi, qj] = q
            break
    else:
        actives.pop()

locations = np.asarray(locs)

# 
Nt = len(locs)
Nc = Nt - Nb


##------------------------------------------------------------------------------
## Chaque capteur c a une énergie energies[c]. Les capteurs du bord du domaine
## (c < Nb) ont au départ une énérgie infinie, les autres ont une énérgie
## distribuée selon une loi exponentielle de moyenne credit. Les capteurs ont
## trois états possibles: réveillé (awake), endormi et mort (energie == 0). Les
## capteurs dans actives sont soit réveillés soit endormis.

# Préparation de l'affichage graphique
plt.ion()
fig = plt.figure()
ax = plt.axes()
ax.axis([-1.05, 1.05, -1.05, 1.05])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


# Initialisation des énergies
it = 0                                    # Numéro de l'itération
total = 0.0                               # Énergie totale perdue par capteur

energies = np.empty(Nt)                   # Les énergies des capteurs
energies[:Nb] = float('inf')
energies[Nb:] = rnd.exponential(credit, Nt-Nb)

awake = np.ones(Nt, dtype='bool')         # Masque des capteurs réveillés

# La bordure du domaine est donnée par une fonction qui retourne True si tout
# les capteurs de sigma sont sur le bord
fence = lambda sigma: all(v < Nb for v in sigma) 


# Construction du complexe des capteurs actifs
neighbors = {i: set() for i in range(Nt)}
faces = defaultdict(set, {(i,): set() for i in range(Nt)})

for u in range(Nt):
    for v in range(u, Nt):
        if fence((u, v)) or abs(locations[u] - locations[v]) < rayon:
            neighbors[u].add(v)
            neighbors[v].add(u)
                
for sigma in flag(neighbors, 4):
    faces[sigma] = set(facets(sigma))


# Simulation de la vie du réseau    
while any(energies[Nb:] > 0.0):
    it = it + 1
    print('\nit:', it)

    # Baisse de l'énergie
    if any(awake):
        energie_step = energies[awake].min()
        total += energie_step
        energies[awake] -= energie_step
    energies[energies < 0.0] = 0.0

    # Mise à jour du complexe
    print('MAJ Complexe')
    faces = { sigma:taus for (sigma,taus) in faces.iteritems() if all(energies[v] > 0.0 for v in sigma) }
    
    # Calcul de l'homologie relative
    print('Homologie')
    H = homology(faces, fence)

    # Fin de vie
    if 2 not in H:
        print('Fin du réseau')
        break
    
    
    # Calcul des nouveaux réveillés
    print('MAJ status')
    newawake = np.zeros(Nt, dtype='bool')
    newawake[:Nb] = True
    if 2 in H and len(H[2]) > 0:
        print('dim H_2: ', len(H[2]))
        for f in H[2][0]:
            newawake[np.asarray(f)] = True

    # Log
    for v in range(Nt):
        if awake[v] and not newawake[v]:
            if energies[v] <= 0.0:
                print('Mort de', v)
            else:
                print('Extinction de', v)
        elif not awake[v] and newawake[v]:
            print('Reveil de', v)

    awake = newawake
        
    # Affichage
    ax.cla()
    ax.set_title('{} - {}'.format(it, total))
    for sigma in H.get(2, [[]])[0]:
        zs = locations[list(sigma)]
        ax.add_artist(Polygon(np.array(zip(zs.real, zs.imag)), fill=True, facecolor='blue', alpha=0.4))
        for tau in facets(sigma):
            zt = locations[list(tau)]
            ax.add_artist(Line2D(zt.real, zt.imag, color='green', lw=2))

    for v, zv in enumerate(locations):
        if awake[v]:
            col='red'
        elif energies[v] > 0.0:
            col='green'
        else:
            col='black'
        ax.text(zv.real, zv.imag, "%2d" % v, fontsize=12, color=col)

    plt.savefig('output-{}-{:03}'.format(seed, it), frameon=False, bbox_inches='tight', pad_inches=0.0)
    plt.draw()

plt.ioff()
