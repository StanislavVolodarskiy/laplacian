
def ParseVerticle(line):
    line = line[2:]
    components = line.split()
    return (float(components[0]), float(components[1]), float(components[2]))

def ParseUv(line):
    line = line[3:]
    components = line.split()
    return (float(components[0]), float(components[1]))

def ParseNormals(line):
    line = line[3:]
    components = line.split()
    return (float(components[0]), float(components[1]), float(components[2]))    

def ParseFace(line, index):
    line = line[2:]
    components = line.split()
    vertices = []
    for i in range(0, len(components)):
        vertices.append(int(components[i].split('/')[index]))
    return vertices

def ParseSingleFaceVertices(line):
    line = line[2:]
    components = line.split(' ')
    vertices = []
    for i in range(0, len(components)):
        vertices.append(int(components[i]))
    return vertices

def ParseFaceVertices(line):    
    return ParseFace(line, 0)

def ParseFaceUvs(line):    
    return ParseFace(line, 1)

def ParseFaceNormals(line):    
    return ParseFace(line, 2)

def LoadObj(path, pivot=[0, 0, 0]):    
    global currentIndex
    lines = []
    with open(path) as f:
        content = f.readlines()
        lines = lines + content

    vertices = []
    faces = []
    uvs = []
    normals = []
    faceUvs = []
    faceNormals = []
    for i in range(0, len(lines)):
        line = lines[i].strip()
        try:            
            if line.startswith('v '):
                v = ParseVerticle(line)
                v = (v[0] - pivot[0], v[1] - pivot[1], v[2] - pivot[2])
                vertices.append(v)
            if line.startswith('f '):
                if '/' in line:
                    faces.append(ParseFaceVertices(line))
                    faceUvs.append(ParseFaceUvs(line))
                    faceNormals.append(ParseFaceNormals(line))
                else:
                    faces.append(ParseSingleFaceVertices(line))
        except Exception as e:
            print('line "' + line + '"')
            raise e

        if line.startswith('vt '):
            uvs.append(ParseUv(line))
        if line.startswith('vn '):
            normals.append(ParseNormals(line))
    return { 'vertices' : vertices, 'faces' : faces, 'faceUvs' : faceUvs, 'faceNormals' : faceNormals, 'uvs' : uvs, 'normals' : normals }
