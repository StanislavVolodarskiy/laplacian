import math
import numpy as np
from math import sin, cos, acos, sqrt

def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        v = tuple(n / mag for n in v)
    return np.array(v)

def Quaternion_toEulerianAngle(w, x, y, z):
    ysqr = y*y
    
    t0 = +2.0 * (w * x + y*z)
    t1 = +1.0 - 2.0 * (x*x + ysqr)
    X = math.degrees(math.atan2(t0, t1))
    
    t2 = +2.0 * (w*y - z*x)
    t2 =  1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    Y = math.degrees(math.asin(t2))
    
    t3 = +2.0 * (w * z + x*y)
    t4 = +1.0 - 2.0 * (ysqr + z*z)
    Z = math.degrees(math.atan2(t3, t4))
        
    return X, Y, Z 

class Quaternion:
    def __init__(self, w=0, x=0, y=0, z=0):
        if isinstance(w, (list, tuple, np.ndarray)):
            x = w[1]
            y = w[2]
            z = w[3]
            w = w[0]
        self._val = np.array([w, x, y, z])

    @staticmethod
    def from_axisangle(theta, v):
        theta = theta
        v = normalize(v)

        new_quaternion = Quaternion()
        new_quaternion._axisangle_to_q(theta, v)
        return new_quaternion

    @staticmethod
    def EulerToQuaternion(rx, ry, rz):
        phi = math.radians(rx)
        theta = math.radians(ry)    
        psi = math.radians(rz)

        cHalfPsi=math.cos(0.5*psi);
        sHalfPsi=math.sin(0.5*psi);
        cHalfTheta=math.cos(0.5*theta);
        sHalfTheta=math.sin(0.5*theta);
        cHalfPhi=math.cos(0.5*phi);
        sHalfPhi=math.sin(0.5*phi);

        w = cHalfPhi*cHalfTheta*cHalfPsi + sHalfPhi*sHalfTheta*sHalfPsi;

        x = sHalfPhi*cHalfTheta*cHalfPsi-cHalfPhi*sHalfTheta*sHalfPsi;

        y = cHalfPhi*sHalfTheta*cHalfPsi+ sHalfPhi*cHalfTheta*sHalfPsi;

        z = cHalfPhi*cHalfTheta*sHalfPsi-sHalfPhi*sHalfTheta*cHalfPsi;

        return Quaternion(w, x, y, z)

    def ToEulerAngle(self):
        w, x, y, z = self._val
        return Quaternion_toEulerianAngle(w, x, y, z)

    def _axisangle_to_q(self, theta, v):
        x = v[0]
        y = v[1]
        z = v[2]

        w = cos(theta/2.)
        x = x * sin(theta/2.)
        y = y * sin(theta/2.)
        z = z * sin(theta/2.)

        self._val = np.array([w, x, y, z])

    def __mul__(self, b):
        if isinstance(b, Quaternion):
            return self._multiply_with_quaternion(b)
        elif isinstance(b, (list, tuple, np.ndarray)):            
            if len(b) != 3:
                raise Exception("Input vector has invalid length {len(b)}")
            return self._multiply_with_vector(b)
        else:
            raise Exception("Multiplication with unknown type {type(b)}")

    def _multiply_with_quaternion(self, q2):
        w1, x1, y1, z1 = self._val
        w2, x2, y2, z2 = q2._val
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        result = Quaternion(w, x, y, z)
        return result

    def _multiply_with_vector(self, v):
        q2 = Quaternion(0, v[0], v[1], v[2])
        return (self * q2 * self.get_conjugate())._val[1:]

    def get_conjugate(self):
        w, x, y, z = self._val
        result = Quaternion(w, -x, -y, -z)
        return result

    def __repr__(self):
        theta, v = self.get_axisangle()
        return "((%.6f; %.6f, %.6f, %.6f))"%(theta, v[0], v[1], v[2])

    def get_axisangle(self):
        w, v = self._val[0], self._val[1:]
        theta = acos(w) * 2.0

        return theta, normalize(v)

    def tolist(self):
        return self._val.tolist()

    def vector_norm(self):
        w, v = self.get_axisangle()
        return np.linalg.norm(v)

    def Inverse(self, tolerance = 0.00001):
        v = self.get_conjugate()._val
        mag2 = sum(n * n for n in v)
        if abs(mag2 - 1.0) > tolerance:
            v = tuple(n / mag2 for n in v)

        return Quaternion(v[0], v[1], v[2], v[3])

    def ToTransform(self):
        """
        Transform a unit quaternion into its corresponding rotation matrix (to
        be applied on the right side).

        :returns: transform matrix
        :rtype: numpy array

        """
        w, x, y, z = self._val
        xx2 = 2 * x * x
        yy2 = 2 * y * y
        zz2 = 2 * z * z
        xy2 = 2 * x * y
        wz2 = 2 * w * z
        zx2 = 2 * z * x
        wy2 = 2 * w * y
        yz2 = 2 * y * z
        wx2 = 2 * w * x

        rmat = np.zeros((4, 4), float)
        rmat[0,0] = 1. - yy2 - zz2
        rmat[0,1] = xy2 - wz2
        rmat[0,2] = zx2 + wy2
        rmat[0,3] = 0
        rmat[1,0] = xy2 + wz2
        rmat[1,1] = 1. - xx2 - zz2
        rmat[1,2] = yz2 - wx2
        rmat[1,3] = 0
        rmat[2,0] = zx2 - wy2
        rmat[2,1] = yz2 + wx2
        rmat[2,2] = 1. - xx2 - yy2
        rmat[2,3] = 0
        rmat[3,0] = 0
        rmat[3,1] = 0
        rmat[3,2] = 0
        rmat[3,3] = 1

        return rmat

    @staticmethod
    def FromTransform(transform):
        """Construct quaternion from the transform/rotation matrix 
        :returns: quaternion formed from transform matrix
        :rtype: numpy array
        """

        t = np.zeros((3, 3), float)

        t[0,0] = transform[0,0]
        t[0,1] = transform[0,1]
        t[0,2] = transform[0,2]
        t[1,0] = transform[1,0]
        t[1,1] = transform[1,1]
        t[1,2] = transform[1,2]
        t[2,0] = transform[2,0]
        t[2,1] = transform[2,1]
        t[2,2] = transform[2,2]

        # Code was copied from perl PDL code that uses backwards index ordering
        T = t.transpose()  
        den = np.array([ 1.0 + T[0,0] - T[1,1] - T[2,2],
                       1.0 - T[0,0] + T[1,1] - T[2,2],
                       1.0 - T[0,0] - T[1,1] + T[2,2],
                       1.0 + T[0,0] + T[1,1] + T[2,2]])

        max_idx = np.flatnonzero(den == max(den))[0]

        q = np.zeros(4)
        q[max_idx] = 0.5 * sqrt(max(den))
        denom = 4.0 * q[max_idx]
        if (max_idx == 0):
            q[1] =  (T[1,0] + T[0,1]) / denom 
            q[2] =  (T[2,0] + T[0,2]) / denom 
            q[3] = -(T[2,1] - T[1,2]) / denom 
        if (max_idx == 1):
            q[0] =  (T[1,0] + T[0,1]) / denom 
            q[2] =  (T[2,1] + T[1,2]) / denom 
            q[3] = -(T[0,2] - T[2,0]) / denom 
        if (max_idx == 2):
            q[0] =  (T[2,0] + T[0,2]) / denom 
            q[1] =  (T[2,1] + T[1,2]) / denom 
            q[3] = -(T[1,0] - T[0,1]) / denom 
        if (max_idx == 3):
            q[0] = -(T[2,1] - T[1,2]) / denom 
            q[1] = -(T[0,2] - T[2,0]) / denom 
            q[2] = -(T[1,0] - T[0,1]) / denom 

        return Quaternion(q[3], q[0], q[1], q[2])

def ToTransform(q):
    w, x, y, z = q
    xx2 = 2 * x * x
    yy2 = 2 * y * y
    zz2 = 2 * z * z
    xy2 = 2 * x * y
    wz2 = 2 * w * z
    zx2 = 2 * z * x
    wy2 = 2 * w * y
    yz2 = 2 * y * z
    wx2 = 2 * w * x

    rmat = np.empty((3, 3), float)
    rmat[0,0] = 1. - yy2 - zz2
    rmat[0,1] = xy2 - wz2
    rmat[0,2] = zx2 + wy2
    rmat[1,0] = xy2 + wz2
    rmat[1,1] = 1. - xx2 - zz2
    rmat[1,2] = yz2 - wx2
    rmat[2,0] = zx2 - wy2
    rmat[2,1] = yz2 + wx2
    rmat[2,2] = 1. - xx2 - yy2

    return rmat

def MultiplyQuaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return [w, x, y, z]

def ApplyRotation(r, v):
    m = ToTransform(r)
    return m.dot(v)

def ApplyRotationToPoints(r, points):
    m = ToTransform(r)    
    return np.tensordot(points, m, axes=([1], [1]))

