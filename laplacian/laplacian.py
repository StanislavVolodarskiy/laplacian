# import numpy as np

import autograd.numpy as np
from autograd import grad


from . import utils


def norm(a):
    return np.sqrt(np.dot(a, a))


def area2(p1, p2, p3):
    a = p1 - p2
    b = p3 - p2
    return norm(np.cross(a, b))


def cot(a, b):
    return np.dot(a, b) / norm(np.cross(a, b))


class Normal(object):
    def __init__(self, dcel, v):
        assert dcel.internal_vertex(v)
        self._neighbours = list(dcel.neighbours(v))
        self._v = v
        self._grads = [grad(lambda v: self(v)[i]) for i in xrange(3)]

    def __call__(self, vertices):
        p = vertices[self._v]

        s = np.zeros(3)
        a2 = 0
        for v1, v2 in utils.pairs(self._neighbours):
            p1 = vertices[v1]
            p2 = vertices[v2]
            s += cot(p - p1, p2 - p1) * (p2 - p) + cot(p - p2, p1 - p2) * (p1 - p)
            a2 += area2(p, p1, p2)
        
        return (1. / (2 * a2)) * s

    def dependencies(self):
        return self._neighbours + [self._v]


class Normals(object):
    def __init__(self, dcel):
        self._normals = [Normal(dcel, v) for v in xrange(dcel.n_vertices) if dcel.internal_vertex(v)]

    def __call__(self, vertices):
        return [n(vertices) for n in self._normals]

    def dependencies(self):
        return [n.dependencies() for n in self._normals]
