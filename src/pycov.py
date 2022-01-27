# -*- coding: utf-8 -*-
from __future__ import print_function

from itertools import count, combinations

import numpy as np
import scipy as sp
import scipy.linalg as lin
import scipy.optimize as opt

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from topology.spaces import CellComplex
from topology.persistence import PairCells
from topology.algorithm import max_cliques

##------------------------------------------------------------------------------
## Paramètres
nsensors = 20                             # Bombre de capteurs
sigma_n = 0.01                            # Variance du bruit de mesure
nsampling = 50                            # Taille de la grille de la regression
pseuil = 0.2                              # seuil \propto varsigma^2
np.random.seed(1235)                      # 1235 => 2 trous pour 20 capteurs

def field(z):
    return 0.7*np.exp(-16*abs(z-0.75-0.5j)**2) + 0.5*np.exp(-16*abs(z-0.2-0.7j)**2);


##------------------------------------------------------------------------------
## Capteurs
class Sensor:
    def __init__(self, id, z, f):
        self.id = id
        self.loc = z
        self.field = f
        self.covar = {}

    def __repr__(self):
        return "%2d@%3f" % (self.id, self.loc)


##------------------------------------------------------------------------------
## Gaussian Process stuff

def kernel(z, w, varsigma, ell):
    z = np.asarray(z)
    w = np.asarray(w)
    if z.shape == ():
        Z, W = z, w
    else:
        Z = z.flatten()[:, np.newaxis].repeat(w.size, axis=1)
        W = w.flatten()[np.newaxis, :].repeat(z.size, axis=0)
    return varsigma**2 * np.exp(-abs(Z-W)**2/ell**2)


def loglikelihood(zm, fm, s, t, n):
    R = lin.cholesky(kernel(zm, zm, s, t) + n * np.eye(zm.size), lower=True)
    return -sum(lin.solve(R, fm) ** 2) - np.log(R.diagonal().prod())


def precision(*vertices):
    K = np.array([[s.covar[t] for s in vertices] for t in vertices])
    ub = K.diagonal()
    lb = K.min(axis=1)
    H = lin.inv(K)
    H = (H + H.T)/2
    x,n,r = opt.fmin_tnc(func=lambda x: x.dot(H.dot(x)),
                         fprime=lambda x: 2*H.dot(x),
                         x0=(lb + ub)/2,
                         bounds=zip(lb, ub),
                         disp=0)
    return x.dot(H.dot(x))


##------------------------------------------------------------------------------
## Champ + Capteur + Estimation

# Création du champ
etendue = np.linspace(0, 1, nsampling)
xe, ye = np.meshgrid(etendue, etendue)
ze = xe + 1j*ye
fe = field(ze)


# Positionnement des capteurs
zm = np.dot([1, 1j], np.random.rand(2, nsensors))
fm = field(zm) + sigma_n * np.random.normal(size=zm.shape)


# Optimisation des hyperparametres pour le GP
varsigma, ell = opt.fmin(lambda x: -loglikelihood(zm, fm, x[0], x[1], sigma_n), [1, 1], disp=False)
print("varsigma:", varsigma, "ell:", ell)


# Création des capteurs
sensors = [Sensor(i, z, f) for i, z, f in zip(count(), zm, fm)]
for s in sensors:
    s.covar = dict((t, kernel(s.loc, t.loc, varsigma, ell)) for t in sensors)
    s.covar[s] = s.covar[s] + sigma_n**2


# Estimation du champ par GP
Kmm = kernel(zm, zm, varsigma, ell) + sigma_n**2 * np.eye(nsensors)
Kem = kernel(ze, zm, varsigma, ell)
Kee = kernel(ze, ze, varsigma, ell)

fs = Kem.dot(lin.solve(Kmm, fm)).reshape(ze.shape)
vs = (Kee - Kem.dot(lin.solve(Kmm, Kem.T))).diagonal().reshape(nsampling, nsampling)


##------------------------------------------------------------------------------
## Construction du complexe
K = CellComplex()
neighbors = {}

# Vertices
for s in sensors:
    K.add_cell(s)


# Edges
for s, t in combinations(sensors, 2):
    if precision(s, t) > pseuil * varsigma**2:
        neighbors.setdefault(s, set()).add(t)
        neighbors.setdefault(t, set()).add(s)
        K.add_cell(frozenset([s, t]), [s, t])


# k-cliques
for clique in max_cliques(neighbors, 4):
    for r in range(3, 5):
        for simplice in combinations(clique, r):
            bords = list(frozenset(s) for s in combinations(simplice, r-1))
            K.add_cell(frozenset(simplice), bords)


##------------------------------------------------------------------------------
## Homologie
H = PairCells(K, lambda x: 0)
holes = [H.cascade(s) for s in K.cells(1) if not H.partner(s)]


##------------------------------------------------------------------------------
# Affichage
plt.figure()

# Field
plt.subplot(221)
plt.imshow(fe, extent=(0,1,0,1), origin='lower')
plt.title('Field')

# Field estimation
plt.subplot(222)
plt.imshow(fs, extent=(0,1,0,1), origin='lower')
plt.plot(zm.real, zm.imag, 'ko')
plt.title('Field estimation')

# Field estimation variance
plt.subplot(223)
plt.imshow(np.log(vs), extent=(0,1,0,1), origin='lower')
plt.title('Field estimation variance (dB)')

# Complex and holes
a = plt.subplot(224, aspect='equal')
plt.title('Complex and coverage holes, WUSPE''s method')
a.axis([0, 1, 0, 1])

for sigma in K.cells(2):
    zs = np.array([(s.loc.real, s.loc.imag) for s in K.vertices(sigma)])
    p = mpatches.Polygon(zs, fill=True, facecolor='b', alpha=.4)
    a.add_artist(p)

for sigma in K.cells(1):
    zs = np.array([s.loc for s in K.vertices(sigma)])
    p = mlines.Line2D(zs.real, zs.imag, color='k')
    a.add_artist(p)

    
for h in holes:
    for e in h:
        zs = np.array([s.loc for s in e])
        p = mlines.Line2D(zs.real, zs.imag, color='r', lw=4, alpha=0.4)
        a.add_artist(p)

plt.show(block=False) # Non-blocking call for ipython
