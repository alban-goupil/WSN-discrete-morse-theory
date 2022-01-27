# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import scipy.linalg as linalg

from itertools import islice


class GP:
    """Représente et simule un champ gaussien variant dans le temps.

    À un instant donné, le champ aux position z et w ont une covariance
    donnée par la fonction kernel. Ce noyau est paramétré par variance,
    spacescale. Le champ est calculé aux position locations. Le
    paramètre timescale permet de calculer la valeur du champs au
    prochain instant. Ce type de champs ne modèlise pas le bruit de
    mesure qui est à prendre en compte autre part.

    Le champs fourni la valeur des paramètres du noyau et la position
    des points de calcul. L'attribut samples contient la valeur du champ
    aux positions données par locations.

    Pour passer à l'instant suivant, un appel à la méthode next suffit.
    Autrement, il est possible d'iterer sur le champ pour obtenir la
    suite de ses valeurs au cours du temps.

    La méthode de simulation est rapide car elle n'est pas exacte. Soit
    X(t) la valeur du champ à l'instant t aux points locations et soit
    X(t+1) celle à l'instant suivant t+1. On a en notant tau=timescale

    [X(t); X(t+dt)] ~ N(0, [K, K*exp(-1/tau^2); K*exp(-1/tau^2), K])

    car le noyau complet avec le temps est homogène est vaut k(x,y) *
    e^(-((tx-ty)/timescale^2). Par conséquent,
    
    X(t+dt) | X(t) ~ N(X(t) * exp(-1/timespace^2), K(1-exp(-1/tau^2))).
    """
    
    def __init__(self, variance, spacescale, timescale, locations):
        """Construit un champ gaussien variant dans le temps."""
        self.variance = variance
        self.spacescale = spacescale
        self.timescale = timescale

        self.locations = locations
        self.covariance = self.kernel(locations, locations)
        self.samples = np.zeros_like(locations)

        self._meancoef = np.exp(-1.0/self.timescale**2)
        D, V = linalg.eig(self.covariance)
        D = np.real(D)
        D[D < 0.0] = 0.0
        A = V.dot(np.diag(np.sqrt(D)))
        self.samples = A.dot(np.random.normal(size=self.samples.size)).reshape(self.samples.shape)
        
        D *= 1.0 - self._meancoef
        self._L = A * np.sqrt(1.0 - self._meancoef)
        D[D > 0.0] = 1.0 / D[D > 0.0]
        self._Q = V.dot(np.diag(D).dot(np.conj(V.T)))
        self.next()

        
    def kernel(self, z, w):
        """retourne la matrice de covariance du noyau entre les points z et w de
        l'espace à un instant donné. Les coefficients sont donnés par le noyau
        variance*exp(-|z-w|^2/spacescale^2)."""
        z = np.asarray(z)
        w = np.asarray(w)
        if z.shape == ():
            Z, W = z, w
        else:
            Z = z.flatten()[:, np.newaxis].repeat(w.size, axis=1)
            W = w.flatten()[np.newaxis, :].repeat(z.size, axis=0)
        return self.variance * np.exp(-np.abs((Z - W) / self.spacescale)**2)


    def __iter__(self):
        while True:
            yield self.next().copy()


    def next(self):
        """Simule le champ à l'instant suivant.

        Simule la valeur du champ aux points locations à l'instant
        suivant t+1 sachant la valeur du champ à ces même points à
        l'instant t."""
        W = np.random.normal(size=self.samples.size) # ~ N(O,I)
        self.samples *= self._meancoef
        self.samples += self._L.dot(W).reshape(self.samples.shape)
        return self.samples

    
    def regression(self, positions, samples=None, noise=None):
        """Fait une regression sur la valeur du champs.

        La méthode retourne la moyenne et la matrice de covariance de la
        valeur du champs aux positions 'positions' sachant la valeurs du
        champs aux positions self.locations. Le bruit de mesure ajouté
        ici est gaussien de variance noise. Si samples vaut None, la
        valeur actuelle du champs aux positions self.locations est
        utilisée sinon l'argument la donne."""
        K = self.covariance
        if samples is None:
            samples = self.samples
        if noise is not None:
            samples = samples.copy() + np.sqrt(noise) * np.random.normal(size=samples.size)
            K = self.covariance.copy()
            for i in range(len(K)):
                K[i,i] = noise
        Kpp = self.kernel(positions, positions)
        Kpl = self.kernel(positions, self.locations)
        KplQ = Kpl.dot(self._Q)
        mu = KplQ.dot(samples.flatten()).reshape(positions.shape)
        cov = Kpp - KplQ.dot(Kpl.T)
        return (mu, cov)

    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Premier exemple: 1D+t
    plt.figure()
    locations = np.linspace(0.0, 1.0, 11)
    field = GP(variance=1.0, spacescale=0.44, timescale=100, locations=locations)
    example = np.asarray(list(islice(field, 200)))
    plt.plot(example)

    # Second exemple: 2D+t
    locations = np.dot([1, 1j], np.random.random((2, 10)))
    etendue = np.linspace(0, 1, 20)
    xe, ye = np.meshgrid(etendue, etendue)
    positions = xe + 1j*ye
    field = GP(variance=1.0, spacescale=0.44, timescale=10, locations=locations)
    for t in range(5):
        field.next()
        if True: # t % 20 == 1:
            [mu, cov] = field.regression(positions, noise=0.001)
            vs = cov.diagonal().reshape(positions.shape)
    
            plt.figure()
            plt.subplot(211)
            plt.imshow(mu, extent=(0,1,0,1), origin='lower')
            plt.plot(field.locations.real, field.locations.imag, 'ko')
            plt.subplot(212)
            plt.imshow(vs, extent=(0,1,0,1), origin='lower')
            plt.plot(field.locations.real, field.locations.imag, 'ko')
            
    
    plt.show()
