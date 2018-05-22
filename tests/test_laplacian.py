from autograd import jacobian
import numpy as np
import scipy.optimize

from laplacian import dcel, laplacian, utils


def p(x, y, z):
    return np.array([x, y, z])


class Vertices(object):
    def __init__(self, x):
        assert len(x.shape) == 1
        assert x.shape[0] % 3 == 0
        self._x = x

    def __len__(self):
        return self._x.shape[0] / 3

    def __getitem__(self, i):
        return self._x[3 * i:3 * (i + 1)]


class Positions(object):
    def __init__(self, dcel):
        self._indices = [v for v in xrange(dcel.n_vertices) if not dcel.internal_vertex(v)]

    def __call__(self, vertices):
        return [vertices[v] for v in self._indices]

    def dependencies(self):
        return [[v] for v in self._indices]


class Systems(object):
    def __init__(self, *systems):
        self._systems = systems

    def __call__(self, vertices):
        result = []
        for f in self._systems:
            result.extend(f(vertices))
        return result

    def dependencies(self):
        result = []
        for f in self._systems:
            result.extend(f.dependencies())
        return result


class TestLaplacian(object):
    def make_disturbed_plane(self, nx, ny, z, disturb):
        def coord(n, i):
            return -n / 2. + i

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
            pp = np.array(p)
            assert pp.shape == (3, )
            vertices[i] = pp + disturb * np.random.rand(*pp.shape)

        return {
            'vertices' : vertices,
            'faces' : faces
        }

    def make_points(self, *seq):
        return [p(*point) for point in seq]

    def make_fan_faces(self, n_points):
        return [(0, v1, v2) for v1, v2 in utils.pairs(xrange(1, n_points))]

    def check_triangle_area2(self, p1, p2, p3, area2):
        np.testing.assert_array_almost_equal(
            laplacian.area2(p(*p1), p(*p2), p(*p3)),
            area2
        )
        
    def check_fan_normal(self, fan_points, expected_normal):
        points = self.make_points(*fan_points)
        faces = self.make_fan_faces(len(fan_points))
        d = dcel.Builder.by_faces(len(points), faces)
        for f in [0.5, 1, 2]:
            np.testing.assert_array_almost_equal(
                laplacian.Normal(d, 0)([f * v for v in points]),
                p(*expected_normal) / f
            )
        
    def optimize_least_squares_on_mesh(
        self,
        faces,
        target_x,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        itol=1e-6,
        etol=1e-6
    ):

        x0 = target_x + 0.2 * np.random.rand(*target_x.shape)

        n_vertices = len(target_x) / 3

        d = dcel.Builder.by_faces(n_vertices, faces)

        system = Systems(Positions(d), laplacian.Normals(d))

        def f(x):
            vertices = Vertices(x)
            return np.array(system(vertices)).flatten()

        target_y = f(target_x)

        jac_sparsity = np.zeros(shape=(len(target_y), len(target_x)), dtype=int)
        deps = system.dependencies()
        assert 3 * len(deps) == len(target_y)
        for i, dep in enumerate(deps):
            for j in dep:
                for ii in range(3 * i, 3 * i + 3):
                    for jj in range(3 * j, 3 * j + 3):
                        jac_sparsity[ii][jj] = 1
        
        result = scipy.optimize.least_squares(
            lambda x: f(x) - target_y,
            x0,
            jac=jacobian(f),
            bounds=(-np.inf, np.inf),
            method='trf',
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            x_scale=1.0,
            loss='linear',
            f_scale=1.0,
            diff_step=None,
            tr_solver='lsmr',
            tr_options={},
            jac_sparsity=jac_sparsity,
            max_nfev=None,
            verbose=0,
            args=(),
            kwargs={}
        )

        assert result.success

        return d, result

    def check_optimize_least_squares_on_mesh(
        self,
        faces,
        target_x,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        itol=1e-6,
        etol=1e-6
    ):
        d, result = self.optimize_least_squares_on_mesh(
            faces,
            target_x,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            itol=itol,
            etol=etol
        )

        # assert result.njev > 1
        # assert result.nfev > 1

        rx = Vertices(result.x)
        tx = Vertices(target_x)
        try:
            for v in range(d.n_vertices):
                if d.internal_vertex(v):
                    n = laplacian.Normal(d, v)
                    rv = n(rx)
                    tv = n(tx)
                    tol = itol
                else:
                    rv = rx[v]
                    tv = tx[v]
                    tol = etol
                assert np.abs(rv - tv).max() < tol
        except AssertionError:
            print('v', v, rx[v], tx[v])
            print('status', result.status)
            print('optimality', result.optimality)
            print('cost', result.cost)
            print('njev', result.njev)
            print('nfev', result.nfev)
            print('grad', result.grad)

            print(result.x)
            print(target_x)
            print(result.x - target_x)
            raise

    def optimize_least_squares_on_plane(self, n):
        mesh = self.make_disturbed_plane(n, n, np.random.rand(), 0.1)

        target = np.array(mesh['vertices']).flatten()
        faces = mesh['faces']

        self.optimize_least_squares_on_mesh(
            faces,
            target,
            ftol=1e-10,
            xtol=1e-12,
            itol=1e-3,
            etol=1e-5
        )

    def test_area2(self):
        self.check_triangle_area2((10, 10, 10), (11, 10, 10), (11, 11, 10), 1)
        self.check_triangle_area2((10, 10, 10), (10, 11, 10), (10, 11, 11), 1)
        self.check_triangle_area2((10, 10, 10), (10, 12, 10), (10, 12, 12), 4)
        self.check_triangle_area2((1, -1, 0), (1, 1, 0), (0, 0, np.sqrt(2)), 2 * np.sqrt(3))

    def test_cot(self):
        assert laplacian.cot(p(1, 0, 0), p(1, 1, 0)) == 1
        assert laplacian.cot(p(1, 0, 0), p(0, 1, 0)) == 0
        assert laplacian.cot(p(1, -1, np.sqrt(2)), p(1, 1, np.sqrt(2))) == 1 / np.sqrt(3)

    def test_normal_plain_fan(self):
        self.check_fan_normal((
            ( 0,  0, 0),
            (-1, -1, 0),
            ( 1, -1, 0),
            ( 1,  1, 0),
            (-1,  1, 0)
        ), (0, 0, 0))

    def test_normal_triangle_pyramids(self):
        # triangle pyramid with right triangles on sides
        self.check_fan_normal((
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1)
        ), (1. / 3., 1. / 3., 1. / 3.))

    def test_normal_symmetric_square_pyramid(self):
        # square pyramid with equilateral triangles as sides
        self.check_fan_normal((
            ( 0,  0, np.sqrt(2)),
            (-1, -1, 0),
            ( 1, -1, 0),
            ( 1,  1, 0),
            (-1,  1, 0)
        ), (0, 0, -np.sqrt(2) / 6))

    def test_scipy_optimize_least_squares_on_fan(self):
        def points(x, y, z):
            return self.make_points(
                (x , y , z),
                (-1, -1, 0),
                ( 1, -1, 0),
                ( 1,  1, 0),
                (-1,  1, 0)
            )

        target = np.array(points(0, 0, 1)).flatten().astype(float)
        target += 0.1 * np.random.rand(*target.shape)
        fan_faces = self.make_fan_faces(len(target) / 3)

        self.check_optimize_least_squares_on_mesh(fan_faces, target)

    def test_scipy_optimize_least_squares_on_plane_2_2(self):
        n = 2
        mesh = self.make_disturbed_plane(n, n, np.random.rand(), 0.1)

        target = np.array(mesh['vertices']).flatten()
        faces = mesh['faces']

        self.check_optimize_least_squares_on_mesh(faces, target, ftol=1e-8, xtol=1e-8, itol=1e-4, etol=1e-5)

    def test_scipy_optimize_least_squares_on_plane_3_3(self):
        n = 3
        mesh = self.make_disturbed_plane(n, n, np.random.rand(), 0.1)

        target = np.array(mesh['vertices']).flatten()
        faces = mesh['faces']

        self.check_optimize_least_squares_on_mesh(
            faces,
            target,
            ftol=1e-10,
            xtol=1e-12,
            itol=1e-3,
            etol=1e-5
        )

    def test_scipy_optimize_least_squares_on_plane_4_4(self):
        self.optimize_least_squares_on_plane(5)

    def test_scipy_optimize_least_squares_on_plane_5_5(self):
        self.optimize_least_squares_on_plane(5)

    def test_scipy_optimize_least_squares_on_plane_6_6(self):
        self.optimize_least_squares_on_plane(6)

    def test_scipy_optimize_least_squares_on_plane_7_7(self):
        self.optimize_least_squares_on_plane(7)

    def test_scipy_optimize_least_squares_on_plane_8_8(self):
        self.optimize_least_squares_on_plane(8)

    def test_scipy_optimize_least_squares_on_plane_9_9(self):
        self.optimize_least_squares_on_plane(9)

    def test_scipy_optimize_least_squares_on_plane_10_10(self):
        self.optimize_least_squares_on_plane(10)

    def test_scipy_optimize_least_squares_on_plane_11_11(self):
        self.optimize_least_squares_on_plane(11)

    def test_scipy_optimize_least_squares_on_plane_12_12(self):
        self.optimize_least_squares_on_plane(12)

    def test_scipy_optimize_least_squares_on_plane_13_13(self):
        self.optimize_least_squares_on_plane(13)

    def test_scipy_optimize_least_squares_on_plane_14_14(self):
        self.optimize_least_squares_on_plane(14)

    def test_scipy_optimize_least_squares_on_plane_20_20(self):
        self.optimize_least_squares_on_plane(20)

    def test_scipy_optimize_least_squares_on_plane_50_50(self):
        self.optimize_least_squares_on_plane(50)

    def test_scipy_optimize_least_squares_on_plane_100_100(self):
        self.optimize_least_squares_on_plane(100)
