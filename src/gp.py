# -*- coding: utf-8 -*-
from __future__ import print_function

from itertools import count, combinations, chain, repeat

import numpy as np
import numpy.random as rnd
import scipy as sp
import scipy.linalg as linalg
import scipy.optimize as opt

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.animation as animation

from topology.spaces import CellComplex
from topology.homology import homology
from topology.algorithm import flag

from gpsim import GP


##------------------------------------------------------------------------------
## Paramètres
gpv = 1                                     # Variance du GP
gps = 0.55                                  # Échelle espace
gpt = 1.0                                   # Échelle temps
gpn = 0.01                                  # Bruit de capteur
Nc = 40                                     # Nombre de capteurs à l'intérieur
Nb = 16                                     # Nombre de capteurs du bord

seuilc = 0.2 * gpv                          # Seuil du critère
nsampling = 50                              # samples pour la régression

rnd.seed(1234)

##------------------------------------------------------------------------------
## Fonctions utilitaire
def rand_location():
    while True:
        z = 2*complex(rnd.random(), rnd.random())-1.0-1.0j
        if np.abs(z) < 1.0:
            return z
        
        
##------------------------------------------------------------------------------
## Les capteurs
class Sensor:
    def __init__(self, id, z):
        self.id = id
        self.location = z
        self.field = None
        self.mean = 0.0
        self.covariance = None   # Covariance spatiale estimée

    def set_neighbors(self, neighbors):
        self.covariance = dict(zip(neighbors, repeat(0.0)))
        
    def update_field(self, f):
        self.field = f

    def update_mean(self, coef=0.01):
        self.mean += coef * (self.field - self.mean)
    
    def update_covariance(self, coef=0.01):
        for other in self.covariance.keys():
            a = (self.field - self.mean) * (other.field - other.mean)
            self.covariance[other] += coef * (a - self.covariance[other])
                    
    def __repr__(self):
        return "%2d@(%3f, %3f)" % (self.id, self.location.real, self.location.imag)


class WSN:
    def __init__(self, Nb, Nc, seuil):
        self.seuil = seuil
        self.Nb = Nb

        # Les capteurs sur le bord du domaine
        self.sensors = [Sensor(i, np.exp(2j * np.pi * i/Nb)) for i in range(Nb)]
        # Les capteurs dans le domaine
        self.sensors.extend(Sensor(i + Nb, rand_location()) for i in range(Nc))

        # Leurs relations de voisinage
        for s in self.sensors:
            s.set_neighbors(self.sensors)

    def __iter__(self):
        return iter(self.sensors)

    def __len__(self):
        return len(self.sensors)

    def are_linked(self, r, s):
        # Test le voisinage sur le bord du domaine
        if r.id < self.Nb and s.id < self.Nb:
            return (s.id - r.id) % self.Nb == 1 or (r.id - s.id) % self.Nb == 1

        # Test de la corrélation 1ere phase
        if r.covariance[s] < self.seuil:
            return False
        
        # Test de la corrélation
        K = np.array([[r.covariance[r], r.covariance[s]],
                      [s.covariance[r], s.covariance[s]]])
        Q = linalg.pinv(K)
        Q = (Q + Q.T)/2
        ub = K.diagonal()
        lb = K.min(axis=1)
        xf, ff, its, imode, smode = opt.fmin_slsqp(func=lambda x: x.dot(Q.dot(x)),
                                                   fprime=lambda x: 2*Q.dot(x),
                                                    x0 = (lb + ub)/2,
                                                    bounds=np.vstack((lb, ub)).T,
                                                    full_output=True,
                                                    disp=0)
        if imode != 0:
            print('Oops:', vertices, '-', smode)
            return float('-inf')
        return ff


    def update(self, samples):
        # Update the measurements
        for s, f in zip(self.sensors, samples):
            s.update_field(f)
            s.update_mean()
        for s in self.sensors:
            s.update_covariance()
            
        # Update the neighborhood relation
        neighbors = {s: set() for s in self.sensors}
        values = {}
        for (sigma, tau) in combinations(self.sensors, 2):
            val = self.are_linked(sigma, tau)
            if val is False: continue
            if val is True: val = float('-inf')
            neighbors[sigma].add(tau)
            neighbors[tau].add(sigma)
            values[frozenset([sigma, tau])] = val

        # Ajoute les connections de corrélation
        K = CellComplex()
        for simplex in flag(neighbors, 3):
            if len(simplex) == 1:
                K.add_cell(list(simplex)[0], value=float('-inf'))
            elif len(simplex) == 2:
                val = values[frozenset(simplex)]
                bord = list(simplex)
                K.add_cell(simplex, bord, value=val)
            else:
                bord = [frozenset(simplex - set([v])) for v in simplex]
                K.add_cell(simplex, bord, value=float('+inf'))


        ## Calcul de l'homologie
        self.complex = K
        self.homology = homology(K)
#        self.homology = homology(K)

    
wsn = WSN(Nb, Nc, seuilc)
    
##------------------------------------------------------------------------------
## Gaussian Process

# Le champs
field = GP(variance=gpv, spacescale=gps, timescale=gpt, locations=[s.location for s in wsn])

# La précision du champs
etendue = np.linspace(-1.0, 1.0, nsampling)
xe, ye = np.meshgrid(etendue, etendue)
positions = xe + 1j*ye
field.next()
[mu, cov] = field.regression(positions, noise=gpn)
pfield = cov.diagonal().reshape(positions.shape)


##------------------------------------------------------------------------------
## Passage d'un instant à l'autre
eqm = []
modif = []
cycles = set()

def step(field, wsn):
    # Mise à jour du champ de données du WSN
    samples = field.next()
    samples += rnd.normal(size=samples.size, scale=gpn**0.5)

    # Mise à jour du WSN
    wsn.update(samples)

    # Calcul de l'eqm
    n = len(wsn)
    eqm.append(linalg.norm(np.array([[u.covariance[v] for u in wsn] for v in wsn]) - field.covariance)/(n*n))

    # Calcul des modifications
    

##------------------------------------------------------------------------------
##  Animation

fig = plt.figure()
a = plt.subplot(211)
b = plt.subplot(212)

def animate(i, field, wsn, a, b):
    step(field, wsn)

    a.cla()
    a.imshow(pfield, extent=(-1,1,-1,1), origin='lower')
    a.set_title(str(i))

    b.clear()
    b.plot(10.0*np.log10(eqm))
    
    K = wsn.complex
    H = wsn.homology
    
    for sigma in K.cells(0):
        a.text(sigma.location.real, sigma.location.imag, "%2d" % sigma.id, fontsize=12, color='red')

    if K.dimension() > 0:
        for sigma in K.cells(1):
            zs = np.array([s.location for s in K.vertices(sigma)])
            p = mlines.Line2D(zs.real, zs.imag, color='green', lw=1)
            a.add_artist(p)

    if K.dimension() > 1:
        for sigma in K.cells(2):
            zs = np.array([(s.location.real, s.location.imag) for s in K.vertices(sigma)])
            p = mpatches.Polygon(zs, fill=True, facecolor='blue', alpha=0.2)
            a.add_artist(p)

    if 1 in H:
        for cycle in H[1]:
            for e in cycle:
                zs = np.array([s.location for s in e])
                p = mlines.Line2D(zs.real, zs.imag, color='red', lw=4, alpha=0.4)
                a.add_artist(p)

    return [a, b]
    
anim = animation.FuncAnimation(fig, animate, fargs=(field, wsn, a, b), blit=False)

plt.show()
