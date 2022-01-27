# -*- coding: latin-1 -*-
from __future__ import print_function

from itertools import count, combinations, chain

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
nsensors = 20                             # Nombre de capteurs
sigma_n = 0.01                            # Variance du bruit de mesure
nsampling = 50                            # Taille de la grille de la regression
pseuil = 0.4                              # seuil \propto varsigma^2
np.random.seed(1235)                      # 1235 => 2 trous pour 20 capteurs

def field(z):
    return 0.7*np.exp(-16*abs(z-0.75-0.5j)**2) + 0.5*np.exp(-16*abs(z-0.2-0.7j-0.5j )**2);


##------------------------------------------------------------------------------
## Capteurs
class Sensor:
    def __init__(self, id, z, f):
        self.id = id
        self.loc = z
        self.field = f
        self.covar = {}

    def __repr__(self):
        return "%2d@(%3f, %3f, %3f)" % (self.id, self.loc.real, self.loc.imag1, self.loc.imag2)


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
    R = lin.cholesky(kernel(zm, zm, s, t) + n * np.eye(zm.size), lower=True, overwrite_a=True)
    return -sum(lin.solve_triangular(R, fm, lower=True) ** 2) - np.log(R.diagonal().prod())


def precision(vertices):
    K = np.array([[s.covar[t] for s in vertices] for t in vertices])
    Q = lin.inv(K)
    Q = (Q + Q.T)/2
    ub = K.diagonal()
    lb = K.min(axis=1)

    xf, ff, its, imode, smode = opt.fmin_slsqp(func=lambda x: x.dot(Q.dot(x)),
                                               fprime=lambda x: 2*Q.dot(x),
                                               x0 = (lb + ub)/2,
                                               bounds=zip(lb, ub),
                                               full_output=True,
                                               disp=0)
    if imode != 0:
        print('Oops:', smode)
        
    return ff


def precision2(vertices, sensors):
    global Kmm, Qmm
#    Kmm = np.array([[s.covar[t] for s in sensors] for t in sensors])
#    Qmm = lin.inv(Kmm)
    ub = Kmm.diagonal()
    lb = [min([s.covar[t] for t in vertices]) for s in sensors]
    
    xf, ff, its, imode, smode = opt.fmin_slsqp(func=lambda x: x.dot(Qmm.dot(x)),
                                               fprime=lambda x: 2*Qmm.dot(x),
                                               x0 = (lb + ub)/2,
                                               bounds=zip(lb, ub),
                                               full_output=True,
                                               disp=0)
    if imode != 0:
        print('Oops:', smode)

    return ff    
                         

##------------------------------------------------------------------------------
## Champ + Capteur + Estimation

# Création du champ
etendue = np.linspace(0, 1, nsampling)
xe, ye, ze1= np.meshgrid(etendue, etendue, etendue)
ze = xe + 1j*ye + 1j*ze1
fe = field(ze)


# Positionnement des capteurs
zm = np.dot([1, 1j, 1j], np.random.rand(3, nsensors))
fm = field(zm) + sigma_n * np.random.normal(size=zm.shape)


# Optimisation des hyperparametres pour le GP
varsigma, ell = opt.fmin(lambda x: -loglikelihood(zm, fm, x[0], x[1], sigma_n), [1, 1], disp=False)
print("varsigma:", varsigma, "ell:", ell)


# Création des capteurs
sensors = [Sensor(i, z, f) for i, z, f in zip(count(), zm, fm)]
for s in sensors:
    s.covar = {t: kernel(s.loc, t.loc, varsigma, ell) for t in sensors}
    s.covar[s] = s.covar[s] + sigma_n**2


# Estimation du champ par GP
Kmm = kernel(zm, zm, varsigma, ell) + sigma_n**2 * np.eye(nsensors)
Qmm = lin.inv(Kmm)
Qmm = (Qmm + Qmm.T) / 2.0
Kem = kernel(ze, zm, varsigma, ell)
Kee = kernel(ze, ze, varsigma, ell)

fs = Kem.dot(lin.solve(Kmm, fm)).reshape(ze.shape)
vs = (Kee - Kem.dot(lin.solve(Kmm, Kem.T))).diagonal().reshape(nsampling, nsampling, nsampling)

vss = vs.copy()
vss[vs > (1-pseuil) * varsigma**2] = float('nan')


##------------------------------------------------------------------------------
## Construction du complexe faÃ§on WUSPE
K = CellComplex()
neighbors = {}

# Vertices
for s in sensors:
    K.add_cell(s)

# Edges
for sigma in combinations(sensors, 2):
    prec = precision2(sigma, sensors)
    # prec = precision(sigma)
    if prec > pseuil * varsigma**2:
        (s, t) = sigma
        neighbors.setdefault(s, set()).add(t)
        neighbors.setdefault(t, set()).add(s)
        K.add_cell(frozenset(sigma), sigma)
        K[frozenset(sigma)]['smin'] = varsigma ** 2 - prec


# Triangles
for sigma in max_cliques(neighbors, 3):
    prec = precision2(sigma, sensors)
    # prec = precision(sigma)
    if prec > pseuil * varsigma**2:
        bords = list(frozenset(s) for s in combinations(sigma, 2))
        K.add_cell(frozenset(sigma), bords)
        K[frozenset(sigma)]['smin'] = varsigma ** 2 - prec

# k-cliques
# for clique in max_cliques(neighbors, 4):
#     for r in range(3, 5):
#         for simplice in combinations(clique, r):
#             bords = list(frozenset(s) for s in combinations(simplice, r-1))
#             K.add_cell(frozenset(simplice), bords)

##------------------------------------------------------------------------------
## Variance vue par les capteurs
vk = np.inf * np.ones(nsampling * nsampling * nsampling)

for sigma in chain(combinations(sensors, 2), max_cliques(neighbors, 3)):
    smin = varsigma **2 - precision2(sigma, sensors)
    for i, z in enumerate(ze.flat):
        # z satisfait la contrainte s'il est plus proche de chaque capteur
        # que le plus éloigné des points de sigma
        if all(abs(u.loc - z) <= max(abs(u.loc - s.loc) for s in sigma) for u in sensors):
#        if all(abs(u.loc - z) < max(abs(u.loc - s.loc) for s in sigma) for u in sigma):
            vk[i] = min(vk[i], smin)
vk = vk.reshape(nsampling, nsampling, nsampling)

##------------------------------------------------------------------------------
## Homologie
H = PairCells(K, lambda x: 0)

##------------------------------------------------------------------------------
# Affichage
plt.figure()

# Field
plt.subplot(321)
plt.imshow(fe, extent=(0,1,0,1), origin='lower')
plt.title('Field')

# Field estimation
plt.subplot(322)
plt.imshow(fs, extent=(0,1,0,1), origin='lower')
plt.plot(zm.real, zm.imag, 'ko')
plt.title('Field estimation')

# Field estimation variance
plt.subplot(323)
plt.imshow(np.log(vs), extent=(0,1,0,1), origin='lower')
plt.title('Field estimation variance (dB)')

# Field estimation variance seuillée
plt.subplot(324)
plt.imshow(np.log(vss), extent=(0,1,0,1), origin='lower')
plt.title('Field estimation variance seuillee (dB)')

# Complex drawing
a = plt.subplot(325, aspect='equal')
plt.title('Complex and coverage holes, WUSPE''s method')
a.axis([0, 1, 0, 1])

# Field estimation variance viewed by complex
plt.subplot(326)
plt.imshow(np.log(vk), extent=(0,1,0,1), origin='lower')
plt.title('Field estimation variance viewed by complex (dB)')

for sigma in K.cells(0):
    a.annotate("%2d" % sigma.id, xy=(0, 0), xytext=(sigma.loc.real, sigma.loc.imag))

for sigma in K.cells(1):
    zs = np.array([s.loc for s in K.vertices(sigma)])
    p = mlines.Line2D(zs.real, zs.imag, color='k')
    a.add_artist(p)

for sigma in K.cells(2):
    zs = np.array([(s.loc.real, s.loc.imag) for s in K.vertices(sigma)])
    p = mpatches.Polygon(zs, fill=True, facecolor='b', alpha=.4)
    a.add_artist(p)

for h in K.cells(1):
    if not H.partner(h):
        for e in H.cascade(h):
            zs = np.array([s.loc for s in e])
            p = mlines.Line2D(zs.real, zs.imag, color='r', lw=4, alpha=0.4)
            a.add_artist(p)
    
plt.show(block=True) # Non-blocking call for ipython
