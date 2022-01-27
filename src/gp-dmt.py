# -*- coding: utf-8 -*-
from __future__ import print_function

from collections import Counter, defaultdict, deque

import numpy as np
import numpy.random as rnd
import scipy.linalg as linalg
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

from gpsim import GP


##------------------------------------------------------------------------------
## Paramètres
gpv = 1                               # Variance du GP
gps = 0.40                            # Échelle espace
gpt = 1.0                             # Échelle temps
gpn = 0.01                            # Bruit de capteur

Nc = 50                               # Nombre de capteurs à l'intérieur
Nb = 16                               # Nombre de capteurs du bord
Nt = Nc + Nb                          # Nombre total de capteurs

alpha = 0.005                         # Coefficient du AR(1) de l'estimation
warmup = 0                            # Nombre d'itérations sans homologie

seuilc = 0.09 * gpv                   # Seuil du critère
seuilv = 0.03                         # Seuil d'acceptation de covariance
niterations = 1100                    # Nombres d'itérations
seed = 1235                           # Graine du générateur

dmax = 3                              # Dimension maximale du data-space ou None

rnd.seed(seed)


##------------------------------------------------------------------------------
## Variables

# Capteurs
locs = np.hstack((np.exp(2j*np.linspace(0.0, np.pi*(Nb-1)/Nb, Nb)),   # capteurs
                  np.sqrt(rnd.sample(Nc)) * np.exp(2j*np.pi*rnd.sample(Nc))))
mean = np.zeros(Nt)                                  # Moyennes estimées
cov  = gpn * np.eye(Nt)                              # Covariances estimées
samples = np.empty(Nt)                               # Mesures du champ gaussien


locs[18] = -.5-.5j
locs[28] = (locs[49] + locs[62])/2
locs[38] = -.2-.6j
locs[58] = .1+.45j
locs[51] = .6+.2j
locs[30] +=  .2j

# Réseau de capteurs
neighbors = list(set() for i in range(Nt))                # Voisinage
faces = defaultdict(set, {(i,):set() for i in range(Nt)}) # Complexe simplicial
altitude = {(i,): None for i in range(Nt)}                # Fonction de Morse
vectors = {(i,): (i,) for i in range(Nt)}                 # Champ de vecteurs
criticals = set((i,) for i in range(Nt))                  # Cellules critiques

# Champ gaussien
field = GP(variance=gpv, spacescale=gps, timescale=gpt, locations=locs)


##------------------------------------------------------------------------------
## Fonctions

def are_linked(cov, r, s):
    # if r < Nb and s < Nb:                                 # Bord du domaine ?
    #     return (s - r) % Nb == 1 or (r - s) % Nb == 1
    if cov[r, s] < seuilv:                   # Test de la corrélation 1ère phase
        return False

    # Test de la corrélation
    K = np.array([[cov[r,r], cov[r,s]], [cov[s,r], cov[s, s]]])
    Q = linalg.pinv(K)
    Q = (Q + Q.T)/2
    ub = K.diagonal()
    lb = K.min(axis=1)

    res = minimize(fun=lambda x, Q: x.dot(Q.dot(x)), x0=(lb + ub)/2, args=(Q,),
                   jac=lambda x, Q: 2*Q.dot(x), method='SLSQP', bounds=zip(lb, ub))

    if not res.success:
        print('Oops:', (r, s), '-', res.message)
        return False
    return res.fun > seuilc


def flag(S, N, k=None):
    """Simplices contenant S de dimension maximum k du flag donné par la
    relation de voisinage N.
    """
    P = set(N[S[0]])
    for s in S: P &= N[s]
    q = deque([(frozenset(S), P)])

    while q:
        R, P = q.popleft()

        if len(R) == k: continue
        while P:
            v = P.pop()
            r = R | {v}
            yield tuple(sorted(r))
            q.append((r, P & set(N[v])))


def facets(cell):
    """Facets of a simplicial cell; e.g. list(facets((1,2,3))) == [(2,3), (1,3), (1,2)]."""
    for i in range(len(cell)):
        yield cell[:i] + cell[(i+1):]


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

    H = defaultdict(list)
    for (z, dz) in alive:
        assert not dz
        cycle = list(z)
        H[len(cycle[0])-1].append(cycle)

    return H


def boundary(faces, chain):
    """Binary boundary of the chain given the complex by faces' relation."""
    dchain = set()
    for sigma in chain:
        dchain ^= faces[sigma]
    return dchain


## Flow I + dV + Vd
def flowstep(faces, V, c):
    """Move the chain c along the vector field V on the complex given by faces."""
    chain = set(c) # I
    chain.symmetric_difference_update(V[tau] for tau in boundary(faces, chain) if len(V[tau]) > len(tau))  # + Vd
    chain.symmetric_difference_update(boundary(faces, (V[tau] for tau in chain if len(V[tau]) > len(tau)))) # + dV
    return chain


##------------------------------------------------------------------------------
## Simulation
eqm = []

plt.ion()
fig = plt.figure()
ax = plt.subplot(211)
ax.axis([-1.05, 1.05, -1.05, 1.05])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

bx = plt.subplot(212)

# Arrière plan: précision du champs
nsampling = 50
etendue = np.linspace(-1.05, 1.05, nsampling)
xe, ye = np.meshgrid(etendue, etendue)
positions = xe + 1j*ye
field.next()
[mu, K] = field.regression(positions, noise=gpn)
pfield = K.diagonal().reshape(positions.shape)


for it in range(warmup):
    # Mise à jour des données capteurs
    samples = field.next() + rnd.normal(size=Nt, scale=gpn**0.5) # Mesures bruitées
    mean += alpha * (samples - mean)                             # MAJ de la moyenne
    cov += alpha * (np.outer(samples - mean, samples - mean) - cov) # et la covariance
    eqm.append(linalg.norm(cov - field.covariance) / cov.size) # EQM de l'estimation

#cov = field.covariance.copy() + gpn*np.eye(Nt)
for it in range(warmup, niterations):
    print('iteration: ', it)

    # Mise à jour des données capteurs
    samples = field.next() + rnd.normal(size=Nt, scale=gpn**0.5) # Mesures bruitées
    mean += alpha * (samples - mean)                             # MAJ de la moyenne
    cov += alpha * (np.outer(samples - mean, samples - mean) - cov) # et la covariance
    eqm.append(linalg.norm(cov - field.covariance) / cov.size) # EQM de l'estimation

    # Mise à jour du voisinage
    newcells = set()
    oldcells = set()
    for u in range(Nt):
        for v in range(u+1, Nt):
            linked = are_linked(cov, u, v)
            waslinked = u in neighbors[v]

            if linked and not waslinked:
                neighbors[u].add(v)
                neighbors[v].add(u)

                if altitude[(u,)] is None: altitude[(u,)] = (it, (u,))
                if altitude[(v,)] is None: altitude[(v,)] = (it, (v,))

                newcells.add((u, v))
                newcells.update(uvw for uvw in flag((u, v), neighbors, dmax))

            elif not linked and waslinked:
                if (u, v) == (47, 61):
                    pass

                oldcells.add((u, v))
                oldcells.update(uvw for uvw in flag((u, v), neighbors, dmax))

                neighbors[u].remove(v)
                neighbors[v].remove(u)

                if not neighbors[u]: altitude[(u,)] = None
                if not neighbors[v]: altitude[(v,)] = None

    if not oldcells and not newcells:
        # Rien de neuf, on recommence
        continue

    # Mise à jour du complexe cellulaire
    for cell in newcells:
        faces[cell].update(facets(cell))
        altitude[cell] = (it, cell)
        vectors[cell] = cell
        criticals.add(cell)

    for cell in oldcells:
        del faces[cell]
        del altitude[cell]
        Vcell = vectors.pop(cell)
        criticals.discard(cell)
        if Vcell not in oldcells:
            vectors[Vcell] = Vcell
            criticals.add(Vcell)

    # Mise à jour du champ de vecteurs discrets.
    potentials = set(newcells)            # Cellules potentiellement associables
    potentials.update(tau for cell in newcells for tau in facets(cell) if tau in criticals)
    potentials.update(tau for cell in oldcells for tau in facets(cell) if tau in criticals)
    potentials -= oldcells                # Pour éviter les cellules qui font un coucou

    while potentials:
        tau = potentials.pop()
        for sigma in faces[tau]:
            if sigma not in potentials: continue

            vectors[sigma] = tau
            vectors[tau] = sigma

            if all(altitude[sigma] >= altitude[s] for s in flowstep(faces, vectors, [sigma])):
                criticals.discard(sigma)
                criticals.discard(tau)
                potentials.discard(sigma)
                break
            else:
                vectors[sigma] = sigma
                vectors[tau] = tau

    # Calcul du complexe de Morse
    morse = {}
    for sigma in criticals:
        prev = set()
        chain = set([sigma])
        while prev != chain:
            prev, chain = chain, flowstep(faces, vectors, chain)

        morse[sigma] = set(boundary(faces, chain) & criticals)

        if len(morse[sigma]) == 1:
            # Une simplification par "critical cancelling" est possible ici!
            pass

    # Calcul de l'homologie
    H = homology(morse)

    # Affichage de l'homologie du complexe
    Cp = Counter(len(c)-1 for c in criticals)       # Taille du complexe de Morse
    Dp = Counter(len(c)-1 for c in faces)           # Taille du data-space

    print('Morse Complex homology')
    print('  k H_k <= M_k <= C_k')
    for k in Dp.iterkeys():
        print('{:3d} {:3d}    {:3d}    {:3d}'.format(k, len(H[k]), Cp[k], Dp[k]))


    # Affichage graphique
    ax.cla()
    ax.imshow(pfield, extent=(-1.05, 1.05, -1.05, 1.05), origin='lower')
    ax.set_title(str(it))
    for sigma in faces.iterkeys():
        zs = locs[list(sigma)]
        if len(sigma) == 1:
            ax.text(zs.real, zs.imag, "%2d" % sigma[0], fontsize=12, color='red')
        elif len(sigma) == 2:
            ax.add_artist(Line2D(zs.real, zs.imag, color='green', lw=2))
        elif len(sigma) == 3:
            ax.add_artist(Polygon(np.array(zip(zs.real, zs.imag)), fill=True, facecolor='blue', alpha=0.4))

    for cycle in H.get(1, []):
        # Calcul du représentant du cycle dans le data space
        prevdatacycle = set()
        datacycle = set(cycle)
        while datacycle <> prevdatacycle:
            prevdatacycle, datacycle = datacycle, flowstep(faces, vectors, datacycle)

        for sigma in datacycle:
            zs = locs[list(sigma)]
            ax.add_artist(Line2D(zs.real, zs.imag, color='red', lw=4, alpha=0.4))

    bx.clear()
    bx.set_xlim(0, niterations)
    bx.plot(10*np.log10(eqm))

    plt.draw()

plt.ioff()
plt.show()
