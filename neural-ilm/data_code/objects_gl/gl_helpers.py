import torch
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

quadratic = gluNewQuadric()
gluQuadricNormals(quadratic, GLU_SMOOTH)

def set_color(rgb):
    r, g, b = rgb
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [r, g, b, 1.])
    glMaterialfv(GL_FRONT, GL_AMBIENT, [r, g, b, 1.])

def square(x, z, dx, dz):
    glVertex3fv((x, 0, z))
    glVertex3fv((x + dx, 0, z))
    glVertex3fv((x + dx, 0, z + dz))
    glVertex3fv((x, 0, z + dz))

def ground():
    c1 = (torch.rand(3) * 0.3 + 0).tolist()
    c2 = (torch.rand(3) * 0.3 + 0).tolist()
    for x in range(-2, 2):
        for z in range(-1, 5):
            if (x + z) % 2 == 0:
                set_color(c1)
            else:
                set_color(c2)
            glBegin(GL_QUADS)
            glNormal3fv([0, -1, 0])
            square(x * 2, - z * 2, 2, -2)
            glEnd()

cube_vertices = (
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 0),
    (1, 0, 1),
    (1, 1, 1),
    (0, 0, 1),
    (0, 1, 1)
    )
cube_surfaces = (
    # (3, 2, 1, 0),
    (0, 1, 2, 3),
    (3, 2, 7, 6),
    (6, 7, 5, 4),
    (4, 5, 1, 0),
    (1, 5, 7, 2),
    (4, 0, 3, 6)
    )
cube_normals = [
    [0, 0, 1], # back face
    [1, 0, 0], # left face
    [0, 0, -1], # front face
    [-1, 0, 0], # right face
    [0, -1, 0], # top face
    [0, 1, 0], # bottom face
]
def _cube(size):
    # glColor3fv((1, 1, 1))
    glBegin(GL_QUADS)
    for i, surface in enumerate(cube_surfaces):
        glNormal3fv(cube_normals[i])
        for j in range(len(surface) - 1, -1, -1):
            vertex = surface[j]
            _x, _y, _z = cube_vertices[vertex]
            glVertex3fv((_x * size - size / 2, _y * size - size / 2, _z * size - size / 2))
    glEnd()

def _sphere(radius):
    glFrontFace(GL_CW)
    gluQuadricOrientation(quadratic, GLU_INSIDE)
    gluSphere(quadratic, radius,32,32)
    glFrontFace(GL_CCW)

def _cylinder(height, radius):
    glFrontFace(GL_CW)
    glPushMatrix()
    glRotatef(-90, 1, 0, 0)
    gluQuadricOrientation(quadratic, GLU_OUTSIDE)
    gluDisk(quadratic, 0, radius, 32, 32)
    gluQuadricOrientation(quadratic, GLU_INSIDE)
    gluCylinder(quadratic,radius,radius,height,32,32)
    glTranslatef(0, 0, height)
    glRotatef(180, 1, 0, 0)
    gluQuadricOrientation(quadratic, GLU_OUTSIDE)
    gluDisk(quadratic, 0, radius, 32, 32)
    glPopMatrix()
    glFrontFace(GL_CCW)

def cube(xyz, size, vertrot):
    x, z, y = xyz
    z = -z
    glPushMatrix()
    glTranslatef(x, y + size / 2, z)
    glRotatef(vertrot, 0, 1, 0)
    _cube(size)
    glPopMatrix()

def sphere(xyz, diameter):
    x, z, y = xyz
    z = -z
    radius = diameter / 2
    glPushMatrix()
    glTranslatef(x, y + radius, z)
    _sphere(radius)
    glPopMatrix()

def cylinder(xyz, height, diameter):
    x, z, y = xyz
    z = -z
    radius = diameter / 2
    glPushMatrix()
    glTranslatef(x, y, z)
    _cylinder(height, radius)
    glPopMatrix()
