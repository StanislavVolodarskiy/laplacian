from . import utils

class _Vertex(object):
    def __init__(self):
        self.outgoing = None


class _Face(object):
    def __init__(self):
        self.edge = None


class _Edge(object):
    def __init__(self):
        self.begin = None
        self.end = None
        self.twin = None
        self.next = None
        self.prev = None
        self.face = None


class Dcel(object):
    def __init__(self, vertices, faces, edges):
        self._vertices = vertices
        self._faces = faces
        self._edges = edges

    @property
    def n_vertices(self):
        return len(self._vertices)

    @property
    def n_faces(self):
        return len(self._faces)

    @property
    def n_edges(self):
        return len(self._edges)

    def outgoing(self, v):
        assert 0 <= v < len(self._vertices)
        return self._vertices[v].outgoing

    def edge(self, f):
        assert 0 <= f < len(self._faces)
        return self._faces[f].edge

    def begin(self, e):
        assert 0 <= e < len(self._edges)
        return self._edges[e].begin

    def end(self, e):
        assert 0 <= e < len(self._edges)
        return self._edges[e].end

    def twin(self, e):
        assert 0 <= e < len(self._edges)
        return self._edges[e].twin

    def next(self, e):
        assert 0 <= e < len(self._edges)
        return self._edges[e].next

    def prev(self, e):
        assert 0 <= e < len(self._edges)
        return self._edges[e].prev

    def face(self, e):
        assert 0 <= e < len(self._edges)
        return self._edges[e].face

    def check(self):
        for v in range(len(self._vertices)):
            assert self.begin(self.outgoing(v)) == v

        for f in range(len(self._faces)):
            assert self.face(self.edge(f)) == f

        for e in range(len(self._edges)):
            assert self.prev(self.next(e)) == e
            assert self.next(self.prev(e)) == e
            assert self.end(e) == self.begin(self.next(e))
            assert self.face(e) == self.face(self.next(e))

        for e1 in range(len(self._edges)):
            e2 = self.twin(e1)
            if e2 is not None:
                assert self.twin(e2) == e1
                assert self.begin(e1) == self.end(e2)

    def internal_vertex(self, v):
        o = self.outgoing(v)
        if o is None:
            return False
        e = o
        while True:
            e = self.twin(self.prev(e))
            if e is None:
                return False
            if e == o:
                return True

    def neighbours(self, v):
        e = o = self.outgoing(v)
        while True:
            yield self.end(e)
            e = self.twin(self.prev(e))
            if e == o:
                break


class Builder(Dcel):
    def __init__(self):
        super(Builder, self).__init__(vertices=[], faces=[], edges=[])

    def dcel(self):
        return Dcel(vertices=self._vertices, faces=self._faces, edges=self._edges)

    def append_vertex(self):
        self._vertices.append(_Vertex())
        return len(self._vertices) - 1

    def append_face(self):
        self._faces.append(_Face())
        return len(self._faces) - 1

    def append_edge(self):
        self._edges.append(_Edge())
        return len(self._edges) - 1

    def set_outgoing(self, v, e):
        assert 0 <= v < len(self._vertices)
        assert 0 <= e < len(self._edges)
        assert self._vertices[v].outgoing is None
        self._vertices[v].outgoing = e

    def set_edge(self, f, e):
        assert 0 <= f < len(self._faces)
        assert 0 <= e < len(self._edges)
        assert self._faces[f].edge is None
        self._faces[f].edge = e

    def set_begin(self, e, v):
        assert 0 <= e < len(self._edges)
        assert 0 <= v < len(self._vertices)
        assert self._edges[e].begin is None
        self._edges[e].begin = v

    def set_end(self, e, v):
        assert 0 <= e < len(self._edges)
        assert 0 <= v < len(self._vertices)
        assert self._edges[e].end is None
        self._edges[e].end = v

    def set_twins(self, e1, e2):
        assert 0 <= e1 < len(self._edges)
        assert 0 <= e2 < len(self._edges)
        assert self._edges[e1].twin is None
        assert self._edges[e2].twin is None
        self._edges[e1].twin = e2
        self._edges[e2].twin = e1

    def set_face(self, e, f):
        assert 0 <= e < len(self._edges)
        assert 0 <= f < len(self._faces)
        assert self._edges[e].face is None
        self._edges[e].face = f

    def set_next_prev(self, ep, en):
        assert 0 <= ep < len(self._edges)
        assert 0 <= en < len(self._edges)
        assert self._edges[ep].next is None
        assert self._edges[en].prev is None
        self._edges[ep].next = en
        self._edges[en].prev = ep

    @classmethod
    def by_faces(cls, n_vertices, faces):
        b = cls()

        for _ in xrange(n_vertices):
            b.append_vertex()

        edge_index = {}

        for f in faces:
            face = b.append_face()

            edges = [b.append_edge() for _ in f]

            for edge in edges:
                b.set_face(edge, face)
            b.set_edge(face, edges[0])

            for edge, (begin, end) in zip(edges, utils.pairs(f)):
                b.set_begin(edge, begin)
                b.set_end(edge, end)

                if b.outgoing(begin) is None:
                    b.set_outgoing(begin, edge)

                assert (begin, end) not in edge_index
                edge_index[(begin, end)] = edge

                twin = edge_index.get((end, begin), None)
                if twin is not None:
                    b.set_twins(edge, twin)

            for p in utils.pairs(edges):
                b.set_next_prev(*p)

        b.check()
        return b.dcel()
