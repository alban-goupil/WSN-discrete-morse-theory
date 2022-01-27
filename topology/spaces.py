# -*- coding: utf-8 -*-

__all__ = [
    'CellComplex',
    'grid_2d'
    ]

class CellComplex:
    def __init__(self):
        self._bags = {}                 # All the cells
        self._cells = [{}]              # Cells ordered by dimension
        self._dimensions = {}
        self._facets = {}
        self._cofacets = {}
        self._vertices = {}
        self._dimension = 0

    def add_cell(self, sigma, facets=[], dim=None, **attr):
        facets = list(facets)
        cofacets = []
        vertices = set()
        
        if dim is None:
            if not facets:
                dimension = 0
            else:
                dimension = 1 + max(self.dimension(tau) for tau in facets)
        else:
            dimension = dim
            
        if dimension == 0:
            vertices.add(sigma)
        else:
            for tau in facets:
                vertices.update(self._vertices[tau])
                self._cofacets[tau].append(sigma)

        self._dimensions[sigma] = dimension
        self._facets[sigma] = facets
        self._cofacets[sigma] = cofacets
        self._vertices[sigma] = vertices

        attr.update({'dimension': dimension, 
                     'facets': facets,
                     'cofacets': cofacets,
                     'vertices': vertices})
        
        if self._dimension < dimension:
            self._cells.extend({} for d in range(dimension - self._dimension))
            self._dimension = dimension
            
        self._bags.setdefault(sigma, {}).update(attr)
        self._cells[dimension].setdefault(sigma, {}).update(attr)

    def remove_cell(self, sigma):
        if self._cofacets[sigma]:
            raise ValueError('cell must have no cofacets')
        for tau in self.facets(sigma):
            self._cofacets[tau].remove(sigma)
        dim = self._dimensions.pop(sigma)
        del self._cells[dim][sigma]
        del self._bags[sigma]
        del self._facets[sigma]
        del self._cofacets[sigma]
        del self._vertices[sigma]
        if dim == self._dimension and not self._cells[dim]:
            self._dimension = self._dimension-1
    
    def cells(self, dim=None, data=False):
        if dim is None:
            if data:
                return self._bags.items()
            return self._bags.keys()
        if data:
            return self._cells[dim].items()
        return self._cells[dim].keys()

    def dimension(self, sigma=None):
        if sigma is None:
            return self._dimension
        return self._dimensions[sigma]

    def facets(self, sigma):
        return self._facets[sigma]

    def cofacets(self, sigma):
        return self._cofacets[sigma]

    def vertices(self, sigma):
        return self._vertices[sigma]

    def star(self, *cells):
        marked = set(*cells)
        queue = list(marked)
        while queue:
            sigma = queue.pop()
            yield sigma
            for tau in self.cofacets(sigma):
                if tau not in marked:
                    queue.append(tau)
                    marked.add(tau)
    
    def __contains__(self, sigma):
        return sigma in self._bags

    def __getitem__(self, sigma):
        return self._bags[sigma]

    def __len__(self):
        return len(self._bags)

def grid_2d(width, height, bounded=False):
    K = CellComplex()
    
    # 0-cells
    for y in range(height):
        for x in range(width):
            K.add_cell((x, y))

    # 1-cells
    for y in range(height):
        for x in range(width):
            if x < width-1:
                K.add_cell(((x, y),(x+1, y)), [(x, y), (x+1, y)])
            if y < height-1:
                K.add_cell(((x, y),(x, y+1)), [(x, y), (x, y+1)])

    # 2-cells
    for y in range(height-1):
        for x in range(width-1):
            K.add_cell(((x, y), (x+1, y), (x, y+1), (x+1, y+1)),
                       [((x, y), (x+1, y)), ((x, y), (x, y+1)),
                        ((x+1, y), (x+1, y+1)), ((x, y+1), (x+1, y+1))])

    # Add the exterior faces if asked
    if bounded:
        faces = []
        faces.extend(((x, 0), (x+1, 0)) for x in range(width-1))
        faces.extend(((x, height-1),(x+1, height-1)) for x in range(width-1))
        faces.extend(((0, y),(0, y+1)) for y in range(height-1))
        faces.extend(((width-1, y),(width-1, y+1)) for y in range(height-1))
        K.add_cell('exterior', faces)

    return K
