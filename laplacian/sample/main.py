import cv2
import numpy
import sys

from cameraprojection import CameraProjection
from utils import LoadDataJson
from meshrenderer import RenderMesh

width = 1024
height = 1024

projection = CameraProjection(width=width, height=height, fl=18)

tensorPath = sys.argv[1]

tensor = LoadDataJson(tensorPath)
assert tensor is not None

tensorMods = numpy.array(tensor['vertices'])
faces = tensor['faces']

image = numpy.zeros((height, width, 3), numpy.uint8)

image[:] = (255, 255, 255)

mesh = { 'vertices' : tensorMods[0] + numpy.array([0, 0, -300]), 'faces' : faces }

RenderMesh(image, projection, mesh)

try:
    cv2.imshow('Image', image)
    cv2.waitKey(0)
finally:
    cv2.destroyAllWindows()
