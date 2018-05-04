import cv2
import numpy
import math
from rotation import RotateVector

def RenderFace(image, projection, vertices, face, color):
    # print(face)
    if len(face) == 3:
        v1 = vertices[face[0] - 1]
        v2 = vertices[face[1] - 1]
        v3 = vertices[face[2] - 1]
        contours = numpy.array([[int(v1[0]), int(v1[1])], [int(v2[0]), int(v2[1])], [int(v3[0]), int(v3[1])]])
    if len(face) == 4:
        v1 = vertices[face[0] - 1]
        v2 = vertices[face[1] - 1]
        v3 = vertices[face[2] - 1]
        v4 = vertices[face[3] - 1]
        contours = numpy.array([[int(v1[0]), int(v1[1])], [int(v2[0]), int(v2[1])], [int(v3[0]), int(v3[1])], [int(v4[0]), int(v4[1])]])
    if len(face) == 5:
        v1 = vertices[face[0] - 1]
        v2 = vertices[face[1] - 1]
        v3 = vertices[face[2] - 1]
        v4 = vertices[face[3] - 1]
        v5 = vertices[face[4] - 1]
        contours = numpy.array([[int(v1[0]), int(v1[1])], [int(v2[0]), int(v2[1])], [int(v3[0]), int(v3[1])], [int(v4[0]), int(v4[1])], [int(v5[0]), int(v5[1])]])

    contours = contours * 16

    # print('Draw face:', contours)
    # cv2.drawContours(image, [contours], 0, (0,0,0), 1, shift=10)
    for i in range(0, len(contours)):
        p1 = contours[i]
        if i == len(contours)-1:
            p2 = contours[0]
        else:
            p2 = contours[i+1]

        assert len(p1) == 2
        assert len(p2) == 2
        cv2.line(image, tuple(p1), tuple(p2), color, 1, cv2.LINE_AA, shift=4)
        cv2.circle(image, tuple(p1), 2, color, thickness=-2, shift=4)
        cv2.circle(image, tuple(p2), 2, color, thickness=-2, shift=4)

def RenderVertices(image, projection, vertices, rotation=None, translation=None, color=(0, 255, 0)):
    if len(vertices) == 0:
        return
    if len(vertices[0]) > 2:
        vertices = projection.ConvertLandmarksTo2d(vertices, rotation=rotation, translation=translation)
    for i in range(0, len(vertices)):
        v1 = vertices[i]
        x = int(v1[0])
        y = int(v1[1])
        cv2.circle(image, (x, y), 3, color, 0)


def IsFaceVisisble(vertices, meshCentroid):
    # if(numpy.dot(numpy.average(vertices, axis=0) - meshCentroid, numpy.array([0, 0, 1])) < 0):
    #     return False
    return True

def RenderMesh(image, projection, mesh, rotation=None, translation=None, color=(0, 0, 0)):
    vertices = numpy.array(mesh['vertices'], float)
    meshCentroid = numpy.average(vertices, axis=0)
    vertices2d = projection.ConvertLandmarksTo2d(vertices, rotation=rotation, translation=translation)
    faces = mesh['faces']
    for i in range(0, len(faces)):
        face = numpy.array(faces[i], int)
        if len(face) < 3:
            continue
        if IsFaceVisisble(vertices[face-1], meshCentroid):
            RenderFace(image, projection, vertices2d, face, color)    

def DrawLandmarks(image, landmarks, color = (255, 0, 0), scale=1.0):
    if isinstance(landmarks[0], (list, numpy.ndarray)):
        for i in range(0, len(landmarks)):
            x = landmarks[i][0] * scale
            y = landmarks[i][1] * scale
            cv2.circle(image, (int(x), int(y)), 2, color, 0)

        return

    for i in range(0, len(landmarks) / 2):
        x = landmarks[i*2+0] * scale
        y = landmarks[i*2+1] * scale
        cv2.circle(image, (int(x), int(y)), 2, color, 0)

def DrawLandmarkLinks(image, landmarks1, landmarks2, count=None, color=(255, 0, 0)):
    if count is None:
        count = len(landmarks1)
    for i in range(0, count):
        l1 = [int(landmarks1[i][0]), int(landmarks1[i][1])]
        l2 = [int(landmarks2[i][0]), int(landmarks2[i][1])]
        cv2.line(image,(l1[0],l1[1]),(l2[0],l2[1]),color,1)

def DrawLandmarksIndexes(image, landmarks, scale=1, color = (255, 255, 255)):
    if scale == None:
        scale = 1
    for i in range(0, len(landmarks)):
        x = landmarks[i][0]
        y = landmarks[i][1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,str(i),(int((x-2)*scale), int(y*scale)), font, 0.3, color, 1)

def RenderLandmarks(image, projection, landmarks3d, scale=1, color = (255, 0, 0), renderNumbers = True, startIndex=0):
    if isinstance(landmarks3d[0], list):
        for i in range(0, len(landmarks3d)):
            lndmrk = (landmarks3d[i][0], landmarks3d[i][1], landmarks3d[i][2])
            x, y = projection.ConvertTo2d(lndmrk)

            # print(x, y)
            if renderNumbers:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image,str(i + startIndex),(int((x-2)*scale), int(y*scale)), font, 0.3, (255,255,255),1)

            cv2.circle(image, (int(x*scale), int(y*scale)), 2, color, 0)
        return
    if isinstance(landmarks3d[0], float):
        for i in range(0, len(landmarks3d) / 3):
            lndmrk = (landmarks3d[i*3+0], landmarks3d[i*3+1], landmarks3d[i*3+2])
            x, y = projection.ConvertTo2d(lndmrk)

            # print(x, y)
            if renderNumbers:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image,str(i + startIndex),(int((x-2)*scale), int(y*scale)), font, 0.3, (255,255,255),1)

            cv2.circle(image, (int(x*scale), int(y*scale)), 2, color, 0)
        return

    for i in range(0, len(landmarks3d)):
        lndmrk = (landmarks3d[i].item(0), landmarks3d[i].item(1), landmarks3d[i].item(2))
        x, y = projection.ConvertTo2d(lndmrk)

        # print(x, y)
        if renderNumbers:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image,str(i + startIndex),(int((x-2)*scale), int(y*scale)), font, 0.3, (255,255,255),1)

        cv2.circle(image, (int(x*scale), int(y*scale)), 2, color, 0)
