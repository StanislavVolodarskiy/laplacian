import numpy
import math

# Axyz - the 3D position of a point A that is to be projected.
# Cxyz - the 3D position of a point C representing the camera.
# Rxyz - the orientation of the camera (represented by Tait-Bryan angles).
# Exyz - the viewer's position relative to the display surface [3] which goes through point C representing the camera.
# Bxy - the 2D projection of A

class CameraProjection:
    def __init__(self, width = 640, height = 640, fl = 35, apertureH = 0.945, rotation = (0, 0, 0), translation = (0, -10, 420), near = 0.100000001, far = 10000):
        self.ScreenWidth = float(width)
        self.ScreenHeight = float(height)
        self.cutMatrix = True
        self.weakProjection = False
        self.apertureH = float(apertureH)

        Rxyz = numpy.array([rotation[0] / 180.0 * math.pi, rotation[1] / 180.0 * math.pi, rotation[2] / 180.0 * math.pi])
        Cxyz = numpy.array([translation[0], translation[1], translation[2], 1])

        # Rotation matrix by X
        M1 = numpy.matrix([[1, 0, 0, 0], [0, math.cos(Rxyz[0]), math.sin(Rxyz[0]), 0], [0, -math.sin(Rxyz[0]), math.cos(Rxyz[0]), 0], [0, 0, 0, 1]])
        # Rotation matrix by Y
        M2 = numpy.matrix([[math.cos(Rxyz[1]), 0, -math.sin(Rxyz[1]), 0], [0, 1, 0, 0], [math.sin(Rxyz[1]), 0, math.cos(Rxyz[1]), 0], [0, 0, 0, 1]])
        # Rotation matrix by Z
        M3 = numpy.matrix([[math.cos(Rxyz[2]), math.sin(Rxyz[2]), 0, 0], [-math.sin(Rxyz[2]), math.cos(Rxyz[2]), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        
        # Translation to camera space
        M5 = numpy.matrix([
            [1, 0, 0, -Cxyz[0]],
            [0, 1, 0, -Cxyz[1]],
            [0, 0, 1, -Cxyz[2]],
            [0, 0, 0, 1]
            ])

        # Projection matrix
        self.Mp = numpy.matrix([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -(far+near) / (near-far), -2*far*near/(near-far)],
                [0, 0, 1, 0]
            ])

        self.SetFocalLength(fl)

        self.Mt = M1 * M2 * M3 * M5
        self.Mi = numpy.linalg.inv(self.Mt)

        cameraRotation = M1 * M2 * M3
        self.cameraLookAtVector = cameraRotation.dot(numpy.array([0.0, 0.0, -1.0, 1.0]))
        self.cameraPoint = Cxyz

        self.weakProjectionScale = 1.0

    def SetFocalLength(self, fl):
        f = (fl / (25.4) * 2.0) / self.apertureH
        self.Mp[0, 0] = f
        self.Mp[1, 1] = f

        self.inverseM = None

        # self.inverseM = numpy.linalg.inv(self.Mp)

    def GetFocalLength(self):
        f = self.Mp[0, 0]
        fl = f * self.apertureH * 25.4 / 2.0
        return fl

    def SetWeakProjectionScale(self, scale):
        self.weakProjectionScale = scale

    def GetWeakProjectionScale(self):
        return self.weakProjectionScale

    def GetWeakProjection(self):
        # M = numpy.matrix([
        #     [self.weakProjectionScale, 0, 0, 0],
        #     [0, self.weakProjectionScale, 0, 0],
        #     [0, 0, 0, 0],
        #     [0, 0, 1, 0]
        #     ])
        M = numpy.matrix([
            [-self.weakProjectionScale, 0, 0, 0],
            [0, -self.weakProjectionScale, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
            ])
        return M

    def GetMatrinxFor2dConvert(self):
        M = self.Mp

        if self.weakProjection:
            # M = numpy.matrix([
            #     [self.weakProjectionScale, 0, 0, 0],
            #     [0, self.weakProjectionScale, 0, 0],
            #     [0, 0, 0, 0],
            #     [0, 0, 1, 0]
            # ])
             M = numpy.matrix([
                [-self.weakProjectionScale, 0, 0, 0],
                [0, -self.weakProjectionScale, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1]
            ])

        if not self.cutMatrix:
            M = self.Mp * self.Mt
        return M

    def ConvertTo2d(self, vec3, size=None):
        Axyz = numpy.array([vec3[0], vec3[1], vec3[2], 1])
         
        M = self.Mp
            
        if self.weakProjection:
            # M = numpy.matrix([
            #     [self.weakProjectionScale, 0, 0, 0],
            #     [0, self.weakProjectionScale, 0, 0],
            #     [0, 0, 0, 0],
            #     [0, 0, 1, 0]
            #     ])
            M = numpy.matrix([
                [-self.weakProjectionScale, 0, 0, 0],
                [0, -self.weakProjectionScale, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1]
            ])

        if not self.cutMatrix:
            M = self.Mp * self.Mt

        Fxyzw = M.dot(Axyz)

        W = Fxyzw.item(3)
        # Projected to [1, -1] [-1, 1], so we need to convert (Maya have right handled coordinates)
        # Bxy = (-Fxyzw.item(0) / W * self.ScreenHeight / 2 + self.ScreenWidth / 2, Fxyzw.item(1) * self.ScreenHeight / 2 / W + self.ScreenHeight / 2)

        if size is None:
            size = [float(self.ScreenWidth), float(self.ScreenHeight)]

        if self.weakProjectionScale:
            Bxy = (-Fxyzw.item(0) * size[1] / 2.0 + size[0], Fxyzw.item(1) * size[1] / 2.0 + size[1])
        else:
            Bxy = (-Fxyzw.item(0) / W * size[1] / 2.0 + size[0] / 2.0, Fxyzw.item(1) / W * size[1] / 2.0 + size[1] / 2.0)
        # Bxy = (-Fxyzw.item(0) / W * size[1] / 2.0 + size[0] / 2.0, Fxyzw.item(1) / W * size[1] / 2.0 + size[1] / 2.0)


        # Bxy = (-Fxyzw.item(0) / W, Fxyzw.item(1) / W)

        return Bxy

    def GetCameraProjectionForPnP(self):
        f = self.Mp[0, 0]
        cameraMatrix = numpy.array([
                                [-f, 0, 0],
                                [0, f, 0],
                                [0, 0, 1]
                                ])
        return cameraMatrix

    def ConvertLandmarksTo2d(self, landmarks, size=None, rotation=None, translation=None):
        landmarks = numpy.insert(landmarks, len(landmarks[0]), 1.0, axis=1)
        M = self.GetMatrinxFor2dConvert()

        Fxyzw = numpy.tensordot(landmarks, M, axes=([1], [1]))
 
        W = Fxyzw[:,3:4]
 
        # Projected to [1, -1] [-1, 1], so we need to convert (Maya have right handled coordinates)
        # Bxy = (-Fxyzw.item(0) / W * self.ScreenHeight / 2 + self.ScreenWidth / 2, Fxyzw.item(1) * self.ScreenHeight / 2 / W + self.ScreenHeight / 2)
 
        if size is None:
            size = numpy.array([float(self.ScreenWidth), float(self.ScreenHeight)])
        size = size / 2.0
 
        landmarks2d = numpy.array([-Fxyzw[:,0:1], Fxyzw[:,1:2]])
        if not self.weakProjection:
            landmarks2d /= W

        landmarks2d[0] = landmarks2d[0] * size[1] + size[0]
        landmarks2d[1] = landmarks2d[1] * size[1] + size[1]
        return numpy.squeeze(numpy.swapaxes(landmarks2d, 0, 1), axis=(2,))

    def ConvertTo3d(self, vec2):
        x = -2.0 * (vec2[0] / self.ScreenWidth - 0.5) 
        y = 2.0 * (vec2[1] / self.ScreenHeight - 0.5)

        Axyz = numpy.array([x, y, 1.0, 1.0])
        Fxyzw = self.inverseM.dot(Axyz)

        point3d = (Fxyzw.item(0) / Fxyzw.item(3) , Fxyzw.item(1) / Fxyzw.item(3), Fxyzw.item(2) / Fxyzw.item(3), Fxyzw.item(3))
        return point3d

    def GetProjectionMatrix(self):
        return self.M

    def GetCameraPoint(self):
        return self.cameraPoint

    def GetCameraLookVector(self):
        return self.cameraLookAtVector

if __name__ == "__main__":

    c = CameraProjection(fov=28)

    v3 = [
        [1, 2, 3],
        [1, 1, 3],
        [1, 1, 1],
        [10, 11, 12],
        ]

    v2 = [
        [321.8004131472575, 298.39504223291016],
        [321.8004131472575, 300.19545538016763],
        [321.79177117029604, 300.2905171267434],
        [338.40356548630297, 281.3525124787638]
        ]

    for i in range(0, len(v3)):
        v = [v3[i][0], v3[i][1], v3[i][2], 1]
        v = c.Mt.dot(v)
        v = [v.item(0) / v.item(3), v.item(1) / v.item(3), v.item(2) / v.item(3)]
        v = c.ConvertTo2d(v)
        numpy.testing.assert_almost_equal(v, v2[i], 6)
        print(v)

