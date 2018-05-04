from laplacian import dcel

class TestDcel(object):
    def make_plane(self, nx, ny, z):
        def coord(n, i):
            return -n / 2. + i

        def crange(n):
            return coord(n, 0), coord(n, n)

        def point(i, j):
            return coord(nx, j), coord(ny, ny - i), z

        points = {}
        faces = []

        def append_vertex(p):
            try:
                i = points[p]
            except KeyError:
                i = points[p] = len(points)
            return i

        def append_face(*points):
            faces.append(tuple(append_vertex(p) for p in points))

        for i in xrange(ny):
            for j in xrange(nx):
                p00 = point(i    , j    )
                p01 = point(i    , j + 1)
                p10 = point(i + 1, j    )
                p11 = point(i + 1, j + 1)
                append_face(p00, p01, p10)
                append_face(p11, p10, p01)

        vertices = [None] * len(points)
        for p, i in points.items():
            assert vertices[i] is None
            vertices[i] = p

        return {
            'vertices' : vertices,
            'faces' : faces,
            'range': (crange(nx), crange(ny))
        }

    def edge_vertex(self, mesh, v):
        for x, r in zip(mesh['vertices'][v], mesh['range']):
            if x in r:
                return True
        return False

    def test_internal_vertex(self):
        m = 5
        n = 10
        mesh = self.make_plane(m, n, -30)
        d = dcel.Builder.by_faces(len(mesh['vertices']), mesh['faces'])
        for v in xrange(d.n_vertices):
            assert not d.internal_vertex(v) == self.edge_vertex(mesh, v)

    def test_neighbours(self):
        d = dcel.Builder.by_faces(4, [[0, 1, 2], [0, 2, 3], [0, 3, 1]])
        assert list(d.neighbours(0)) == [1, 2, 3]
