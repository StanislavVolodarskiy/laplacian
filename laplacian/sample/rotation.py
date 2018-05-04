import numpy
import math
from quaternion import Quaternion, ApplyRotation, ApplyRotationToPoints

def RotateQuaternion(v3, quaternion):
    q = Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])

    v = q * v3

    return (v.item(0), v.item(1), v.item(2))

def RotateVector(v3, rotation):
    if len(rotation) > 3:
        # return RotateQuaternion(v3, rotation)
        return ApplyRotation(rotation, v3)
    Rxyz = (rotation[0] * math.pi / 180.0, rotation[1] * math.pi / 180.0, rotation[2] * math.pi / 180.0)
    v = numpy.matrix((v3[0], v3[1], v3[2]))
    # Rotation matrix by X
    M1 = numpy.matrix([[1, 0, 0], [0, math.cos(Rxyz[0]), math.sin(Rxyz[0])], [0, -math.sin(Rxyz[0]), math.cos(Rxyz[0])]])
    # Rotation matrix by Y
    M2 = numpy.matrix([[math.cos(Rxyz[1]), 0, -math.sin(Rxyz[1])], [0, 1, 0], [math.sin(Rxyz[1]), 0, math.cos(Rxyz[1])]])
    # Rotation matrix by Z
    M3 = numpy.matrix([[math.cos(Rxyz[2]), math.sin(Rxyz[2]), 0], [-math.sin(Rxyz[2]), math.cos(Rxyz[2]), 0], [0, 0, 1]])

    M = M1 * M2 * M3

    v = v * M

    return (v.item(0), v.item(1), v.item(2))

def RotatePoints(points, rotation):
    if len(rotation) > 3:
        return ApplyRotationToPoints(rotation, points)

    Rxyz = (rotation[0] * math.pi / 180.0, rotation[1] * math.pi / 180.0, rotation[2] * math.pi / 180.0)
    # Rotation matrix by X
    M1 = numpy.matrix([[1, 0, 0], [0, math.cos(Rxyz[0]), math.sin(Rxyz[0])], [0, -math.sin(Rxyz[0]), math.cos(Rxyz[0])]])
    # Rotation matrix by Y
    M2 = numpy.matrix([[math.cos(Rxyz[1]), 0, -math.sin(Rxyz[1])], [0, 1, 0], [math.sin(Rxyz[1]), 0, math.cos(Rxyz[1])]])
    # Rotation matrix by Z
    M3 = numpy.matrix([[math.cos(Rxyz[2]), math.sin(Rxyz[2]), 0], [-math.sin(Rxyz[2]), math.cos(Rxyz[2]), 0], [0, 0, 1]])

    M = M1 * M2 * M3

    pointsRotated = np.empty([len(points), 3])
    for i in range(0, len(points)):
        v = numpy.matrix((v3[0], v3[1], v3[2]))
        v = v * M
        pointsRotated[i] = v

    return pointsRotated
