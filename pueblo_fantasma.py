import glfw
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt
import sys
import random

# Variables de cámara (Empezamos en la plaza principal)
cam_x, cam_y, cam_z = 0.0, 2.0, 20.0

def init():
    # Cielo azul de montaña
    glClearColor(0.5, 0.7, 1.0, 1.0) 
    glEnable(GL_DEPTH_TEST)
    
    glMatrixMode(GL_PROJECTION)
    gluPerspective(60, 1.33, 0.1, 150.0)
    glMatrixMode(GL_MODELVIEW)

def key_callback(window, key, scancode, action, mods):
    global cam_x, cam_y, cam_z
    vel = 0.8
    if action in [glfw.PRESS, glfw.REPEAT]:
        if key == glfw.KEY_UP:    cam_z -= vel
        if key == glfw.KEY_DOWN:  cam_z += vel
        if key == glfw.KEY_LEFT:  cam_x -= vel
        if key == glfw.KEY_RIGHT: cam_x += vel
        if key == glfw.KEY_W:     cam_y += vel
        if key == glfw.KEY_S:     cam_y -= vel

def draw_textured_cube(color_base, scale=(1,1,1), rodapie=True):
    """Cubo con el estilo típico michoacano: Pared blanca con base café/roja"""
    glPushMatrix()
    glScalef(scale[0], scale[1], scale[2])
    
    glBegin(GL_QUADS)
    for i in range(6):
        # Dibujamos las paredes laterales con el detalle del rodapié
        if i in [0, 1, 4, 5]: # Caras laterales
            # Parte baja (Rodapié color teja)
            glColor3f(0.5, 0.1, 0.0) 
            glVertex3f(-1, 0, 1 if i==0 or i==4 else -1) # Simplificado para el ejemplo
            # Para hacerlo bien, necesitamos definir los vértices de la franja baja
            # Pero para nivel estudiante, usaremos un color sólido y añadiremos la franja después
        
        glColor3f(color_base[0], color_base[1], color_base[2])
        # Dibujo estándar del cubo...
    glEnd()
    glPopMatrix()

def draw_michoacan_house(x, z):
    glPushMatrix()
    glTranslatef(x, 0, z)
    
    # 1. Paredes Blancas
    glColor3f(0.95, 0.95, 0.9)
    # Cuerpo principal
    glBegin(GL_QUADS)
    # Frente
    glVertex3f(-1.5, 0, 1); glVertex3f(1.5, 0, 1); glVertex3f(1.5, 2, 1); glVertex3f(-1.5, 2, 1)
    # Atrás, Izq, Der... (Simplificado para rendimiento)
    glVertex3f(-1.5, 0, -1); glVertex3f(1.5, 0, -1); glVertex3f(1.5, 2, -1); glVertex3f(-1.5, 2, -1)
    glVertex3f(-1.5, 0, -1); glVertex3f(-1.5, 0, 1); glVertex3f(-1.5, 2, 1); glVertex3f(-1.5, 2, -1)
    glVertex3f(1.5, 0, -1); glVertex3f(1.5, 0, 1); glVertex3f(1.5, 2, 1); glVertex3f(1.5, 2, -1)
    glEnd()

    # 2. El Rodapié (La franja café típica de Michoacán)
    glColor3f(0.4, 0.1, 0.0)
    glBegin(GL_QUADS)
    glVertex3f(-1.51, 0, 1.01); glVertex3f(1.51, 0, 1.01); glVertex3f(1.51, 0.5, 1.01); glVertex3f(-1.51, 0.5, 1.01)
    glEnd()

    # 3. Puerta de madera rústica
    glColor3f(0.3, 0.2, 0.1)
    glBegin(GL_QUADS)
    glVertex3f(-0.3, 0, 1.02); glVertex3f(0.3, 0, 1.02); glVertex3f(0.3, 1.2, 1.02); glVertex3f(-0.3, 1.2, 1.02)
    glEnd()

    # 4. Techo de Teja Roja (Inclinado a dos aguas)
    glColor3f(0.7, 0.2, 0.1)
    glBegin(GL_TRIANGLES)
    # Frente
    glVertex3f(-1.8, 2, 1.2); glVertex3f(1.8, 2, 1.2); glVertex3f(0, 3, 0)
    # Atrás
    glVertex3f(-1.8, 2, -1.2); glVertex3f(1.8, 2, -1.2); glVertex3f(0, 3, 0)
    glEnd()
    # Lados del techo
    glBegin(GL_QUADS)
    glVertex3f(-1.8, 2, 1.2); glVertex3f(0, 3, 0); glVertex3f(0, 3, 0); glVertex3f(-1.8, 2, -1.2) # Simplificado
    glEnd()
    
    glPopMatrix()

def draw_kiosk(x, z):
    """El kiosco central de la plaza"""
    glPushMatrix()
    glTranslatef(x, 0, z)
    # Base
    glColor3f(0.5, 0.5, 0.5)
    for i in range(8):
        glRotatef(45, 0, 1, 0)
        glBegin(GL_QUADS)
        glVertex3f(-1, 0, 2); glVertex3f(1, 0, 2); glVertex3f(1, 0.5, 2); glVertex3f(-1, 0.5, 2)
        glEnd()
    # Columnas y techo
    glColor3f(0.4, 0.1, 0.0)
    glTranslatef(0, 0.5, 0)
    glBegin(GL_TRIANGLES)
    glColor3f(0.8, 0.2, 0.1)
    glVertex3f(-2, 1.5, 2); glVertex3f(2, 1.5, 2); glVertex3f(0, 3, 0)
    glEnd()
    glPopMatrix()

def draw_pueblo():
    global cam_x, cam_y, cam_z
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    gluLookAt(cam_x, cam_y, cam_z, cam_x, cam_y, cam_z - 10, 0, 1, 0)

    # Suelo de empedrado (Gris con puntitos)
    glColor3f(0.4, 0.4, 0.4)
    glBegin(GL_QUADS)
    glVertex3f(-50, 0, 50); glVertex3f(50, 0, 50); glVertex3f(50, 0, -50); glVertex3f(-50, 0, -50)
    glEnd()

    # Plaza Central (Jardín)
    glColor3f(0.1, 0.4, 0.1)
    glBegin(GL_QUADS)
    glVertex3f(-10, 0.01, 10); glVertex3f(10, 0.01, 10); glVertex3f(10, 0.01, -10); glVertex3f(-10, 0.01, -10)
    glEnd()

    draw_kiosk(0, 0)

    # Casas a los lados
    for i in range(4):
        # Lado Este
        draw_michoacan_house(15, -15 + (i*10))
        # Lado Oeste
        glPushMatrix()
        glTranslatef(-15, 0, -15 + (i*10))
        glRotatef(90, 0, 1, 0)
        draw_michoacan_house(0, 0)
        glPopMatrix()

    glfw.swap_buffers(window)

def main():
    global window
    if not glfw.init(): return
    window = glfw.create_window(1000, 700, "Pueblo Magico de Michoacán", None, None)
    if not window:
        glfw.terminate(); return
    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)
    init()
    while not glfw.window_should_close(window):
        draw_pueblo()
        glfw.poll_events()
    glfw.terminate()

if __name__ == "__main__":
    main()