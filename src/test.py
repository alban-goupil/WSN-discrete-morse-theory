# -*- coding: utf-8 -*-
from __future__ import print_function

from itertools import count, combinations

from topology.spaces import CellComplex
from topology.persistence import PairCells

## Construction du complexe de test

K = CellComplex()

# Sommets
K.add_cell('a', value=0)
K.add_cell('b', value=0)
K.add_cell('c', value=0)
K.add_cell('d', value=0)
K.add_cell('e', value=0)

# Arêtes
K.add_cell('ab', 'ab', value=3)
K.add_cell('bc', 'bc', value=3)
K.add_cell('cd', 'cd', value=1)
K.add_cell('de', 'de', value=2)
K.add_cell('ae', 'ae', value=2)
K.add_cell('ad', 'ad', value=1)
K.add_cell('bd', 'bd', value=5)

# Triangles
K.add_cell('ade', ['ad', 'ae', 'de'], value=100)
K.add_cell('bcd', ['bc', 'bd', 'cd'], value=100)

## Calcul de la persistance
H = PairCells(K, lambda x: K[x]['value'])

## Affichage des cycles
cycles = [H.cascade(s) for s in K.cells() if not H.partner(s)]

for cycle in cycles:
    print(cycle)
