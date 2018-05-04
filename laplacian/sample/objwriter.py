

def WriteObj(mesh, path):
    vertices = mesh['vertices']
    faces = mesh['faces']

    with open(path, "w") as file:
        file = open(path, "w")

        file.write("# Vertices count: %d\n" % len(vertices))
        file.write("# Faces count: %d\n" % len(faces))
        file.write("\n")

        for i in range(0, len(vertices)):
            file.write("v %f %f %f\n" % (vertices[i][0], vertices[i][1], vertices[i][2]))

        for i in range(0, len(faces)):
            file.write("f ")
            for j in range(0, len(faces[i])):
                file.write("%d " % faces[i][j])
            file.write("\n")
