import time
import logging
import json
import os
import datetime
import math
import numpy as np
import numpy
import struct
try:
    from clahe import CLAHE
except ImportError:
    pass
finally:
    pass
 
from PIL import Image
import base64
try:
    from skimage import io
    from skimage import img_as_ubyte
except ImportError:
    pass
finally:
    pass

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

def PackFloatToArray(data, offset, val):
    v = numpy.float32(val)
    b = numpy.array([v]).tobytes()

    for i in range(0, len(b)):
        data[offset + i] = b[i]

    return len(b)

def PackHalfFloatToArray(data, offset, val):
    v = numpy.float16(val)
    b = numpy.array([v]).tobytes()

    for i in range(0, len(b)):
        data[offset + i] = b[i]

    return len(b)

def PackIntToArray(data, offset, val):
    v = numpy.int32(val)
    b = numpy.array([v]).tobytes()
    for i in range(0, len(b)):
        data[offset + i] = b[i]

    return len(b)

def IntToBytes(val):
    return struct.pack("i", int(val))

def FloatToBytes(val):
    b = bytearray(4)
    PackFloatToArray(b, 0, val)
    return b

def HalfFloatToBytes(val):
    b = bytearray(2)
    PackHalfFloatToArray(b, 0, val)
    return b


def SetLogFile(filepath):
    logger = logging.getLogger()
    logger.handlers = []
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def PrintEstimate(startTime, current, total, title="Pass"):
    elapsedTime = time.time() - startTime
    p = current * 1.0 / total
    if p == 0:
        totalTime = 0
    else:
        totalTime = (1.0 - p) / p * elapsedTime

    if totalTime == 0:
        estimate = "unknown"
    else:
        estimate = str(datetime.timedelta(seconds=int(totalTime)))
    LogInfo(title + " " + str(current) + " from " + str(total) + ", estimate " + estimate)

def PrintTotalTime(startTime, title="Total"):
    LogInfo(title + ": " + str(datetime.timedelta(seconds=int(time.time() - startTime))))

def LogInfo(m, o=None):
    if not o is None:
        logging.info(str(m) + " " + str(o))
        return
    logging.info(str(m))

def LoadGrayScaleImage(path):
    return Image.open(path).convert('LA')

def LoadImage(path):
    if path is None or not os.path.exists(path):        
        return None
    return Image.open(path)

def LoadImageSkimage(path):
    if path is None or not os.path.exists(path):
        return None
    return img_as_ubyte(io.imread(path, True))

def LoadClaheImage(path):
    grayImg = LoadImageSkimage(path)
    CLAHE(grayImg, 8, 8, 128, 3)
    return grayImg

def DataJsonFromString(string):
    if string is None or len(string) == 0:
        return None
    data = json.loads(string)
    return data

def StringFromJsonData(jsonData):
    if jsonData is None:
        return None
    jsonString = json.dumps(jsonData)
    return jsonString

def LoadDataJson(path):
    if not os.path.exists(path):
        return None
    with open(path) as data_file:
        data = json.load(data_file)
        return data

def WriteDataJson(data, path):    
    with open(path, 'w') as outfile:
        json.dump(data, outfile, indent=2, sort_keys=True)

def Array1d(n, v):
    a = []
    for i in range(0, n):
        a.append(v)
    return a

def Array2d(n, m, v):
    # Slower for some reason!
    #numpy.zeros((P, P), dtype=numpy.int)
    a = []
    for i in range(0, n):
        a.append([])
        for j in range(0, m):
            a[i].append(v)
    return a

def GetPixel(I, x, y):
    if type(I) == numpy.ndarray:
        width = numpy.int32(I.shape[1])
        height = numpy.int32(I.shape[0])

        if x < 0 or x >= width or y < 0 or y >= height:
            return int(0)
        return I[int(y)][int(x)]

    width, height = I.size
    if x < 0 or x >= width or y < 0 or y >= height:
        return int(0)

    return I.getpixel((int(x), int(y)))[0]

def GetBilinearFilteredPixel(I, u, v):
   x = int(math.floor(u))
   y = int(math.floor(v))
   u_ratio = u - x
   v_ratio = v - y
   u_opposite = 1.0 - u_ratio
   v_opposite = 1.0 - v_ratio
   result = (GetPixel(I, x, y) * u_opposite + GetPixel(I, x+1, y) * u_ratio) * v_opposite + (GetPixel(I, x, y+1) * u_opposite  + GetPixel(I, x+1, y+1) * u_ratio) * v_ratio
   return result

def FovFromFocalLength(fl, aperture = 1.417):
    fov = (0.5 * aperture / 0.03937) / fl
    fov = 2.0 * math.atan(fov)
    fov = 57.29578 * fov
    return fov

def FocalLengthFromFov(fov, aperture = 1.417):
    fl = math.tan(0.00872665 * fov)
    fl = (0.5 * aperture / 0.03937) / fl

    return fl

def GetFilesAtPath(path, ext=None, removeExt=False):
    items = os.listdir(path)
    result = []
    for item in items:
        if os.path.isdir(path + item) or (not ext is None and not item.endswith(ext)):
            continue
        if removeExt:
            item = item[:-len(ext)]
        result.append(item)
    return result

def GetFirstNumeratedFile(path, prefix, suffix):
    for i in range(0, 1000):
        filePath = path + prefix + str(i).zfill(4) + suffix
        if os.path.exists(filePath):
            return filePath
        filePath = path + prefix + str(i).zfill(1) + suffix
        if os.path.exists(filePath):
            return filePath
    return None

def LoadFirstPose(basePath):
    filePath = GetFirstNumeratedFile(basePath, "pose_", ".json")
    return LoadDataJson(filePath)

def VectorToStr(x):
    s = "[ "
    for i in range(0, len(x)):
        if i > 0:
            s = s + ", "
        s = s + "{:.2f}".format(x[i])
    s = s + " ]"
    return s

def CompareByPercent(a, b, p):
    change = abs((a - b) / ((a + b) / 2))
    return change < p

def LoadFileBase64(name):
    with open(name, "rb") as file:
        encoded_string = base64.b64encode(file.read())

    return encoded_string

def RandomSelectUniqueIndexes(probablility, n):
    p = numpy.array(probablility)
    p = p / numpy.sum(p)

    r = numpy.random.choice(len(p), size = n, p=p, replace=False)

    return r

#-------------------------------------------------------For debug-------------------------------------------------------#
try:
    from PIL import Image, ImageDraw
except ImportError:
    pass
finally:
    pass

def DrawPoints2d(image, points, color, size):
    draw = ImageDraw.Draw(image)
    for i in range(0, len(points)):
        p = points[i]
        draw.ellipse((p[0] - size,
                      p[1] - size,
                      p[0] + size,
                      p[1] + size),
                     fill=color)

def DrawPoints3d(image, points, color, size, projection):
    p2d = projection.ConvertLandmarksTo2d(points)
    DrawPoints2d(image, p2d, color, size)

def DrawLines(image, points, color, size):
    draw = ImageDraw.Draw(image)
    for i in range(0, len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        draw.line((p1[0], p1[1],
                   p2[0], p2[1]),
                  fill=color,
                  width=size)

def DrawText(image, text, position, color, font = None):
    draw = ImageDraw.Draw(image)
    draw.text(position, text, fill=color, font=font)

def AverageLandmarks(landmarks, dim=2):
    average = None
    count = 0
    for i in range(0, len(landmarks)):
        if average is None:
            average = numpy.zeros(landmarks[i].shape, numpy.float64)

        average = average + landmarks[i]
        count = count + 1

    print("Average count:" + str(count))

    average = average / count

    landmarks = numpy.reshape(numpy.array(average, numpy.float64), (-1, dim))
    if dim < 3:
        landmarks = landmarks - CentroidLandmarks(landmarks)

    return landmarks

def CentroidLandmarks(landmarks):
    return numpy.sum(landmarks, axis=0) / len(landmarks)