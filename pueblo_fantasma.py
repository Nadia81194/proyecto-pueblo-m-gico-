import cv2
import mediapipe as mp
import glfw
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt, gluCylinder, gluSphere, gluNewQuadric
import math
import random

cam_x, cam_y, cam_z = 0.0, 2.5, 28.0
cam_yaw = 0.0          # rotación horizontal en grados
VEL = 0.5

random.seed(42)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
cap = cv2.VideoCapture(0)

def init():
    glClearColor(0.52, 0.73, 1.0, 1.0)   # cielo azul 
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(65, 1000/700, 0.1, 300.0)
    glMatrixMode(GL_MODELVIEW)

def hand_control():
    global cam_x, cam_y, cam_z, cam_yaw

    ok, frame = cap.read()
    if not ok:
        return

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        for lm in hand.landmark:
            x = int(lm.x*w)
            y = int(lm.y*h)

        # índice punta
        dedo = hand.landmark[8]

        px = dedo.x
        py = dedo.y

        rad = math.radians(cam_yaw)

        fx = -math.sin(rad)
        fz = -math.cos(rad)

        rx = math.cos(rad)
        rz = -math.sin(rad)

        # Mano izquierda → girar izquierda
        if px < 0.35:
            cam_yaw -= 3

        # Mano derecha → girar derecha
        elif px > 0.65:
            cam_yaw += 3

        # Mano arriba → avanzar
        if py < 0.35:
            cam_x += fx*VEL
            cam_z += fz*VEL

        # Mano abajo → retroceder
        elif py > 0.65:
            cam_x -= fx*VEL
            cam_z -= fz*VEL

        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

    cv2.imshow("Control con mano", frame)

    if cv2.waitKey(1)==27:
        glfw.set_window_should_close(window,True)


def quad(x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3):
    glVertex3f(x0,y0,z0); glVertex3f(x1,y1,z1)
    glVertex3f(x2,y2,z2); glVertex3f(x3,y3,z3)

def box(x0,y0,z0, x1,y1,z1):
    """Caja sólida de (x0,y0,z0) a (x1,y1,z1)"""
    glBegin(GL_QUADS)
    # Frente
    quad(x0,y0,z1, x1,y0,z1, x1,y1,z1, x0,y1,z1)
    # Atrás
    quad(x1,y0,z0, x0,y0,z0, x0,y1,z0, x1,y1,z0)
    # Izquierda
    quad(x0,y0,z0, x0,y0,z1, x0,y1,z1, x0,y1,z0)
    # Derecha
    quad(x1,y0,z1, x1,y0,z0, x1,y1,z0, x1,y1,z1)
    # Arriba
    quad(x0,y1,z1, x1,y1,z1, x1,y1,z0, x0,y1,z0)
    # Abajo
    quad(x0,y0,z0, x1,y0,z0, x1,y0,z1, x0,y0,z1)
    glEnd()

def cylinder_gl(radius, height, slices=12):
    q = gluNewQuadric()
    gluCylinder(q, radius, radius, height, slices, 1)

def sphere_gl(radius, slices=10):
    q = gluNewQuadric()
    gluSphere(q, radius, slices, slices)

#pino
def draw_tree(x, z, h=3.5, trunk_r=0.15, crown_r=0.9, kind=0):
    glPushMatrix()
    glTranslatef(x, 0, z)

    # Tronco
    glColor3f(0.35, 0.22, 0.08)
    glPushMatrix()
    glRotatef(-90,1,0,0)
    cylinder_gl(trunk_r, h*0.38)
    glPopMatrix()

    base_y = h*0.38

    if kind == 0:  
        glColor3f(0.08, 0.38, 0.08)
        for lvl in range(3):
            r = crown_r * (1.0 - lvl*0.22)
            hy = base_y + lvl*(h*0.20)
            glPushMatrix()
            glTranslatef(0, hy, 0)
            glRotatef(-90,1,0,0)
            cylinder_gl(r, h*0.35, 16)
            glPopMatrix()
    else:           # Frondoso (esfera)
        glColor3f(0.1, 0.45, 0.1)
        glPushMatrix()
        glTranslatef(0, base_y + crown_r*0.85, 0)
        sphere_gl(crown_r, 14)
        glPopMatrix()

    glPopMatrix()


#  BANCA DE PLAZA

def draw_bench(x, z, rot=0):
    glPushMatrix()
    glTranslatef(x, 0, z)
    glRotatef(rot, 0,1,0)
    # Patas
    glColor3f(0.25, 0.15, 0.05)
    for px in [-0.55, 0.55]:
        box(px-0.07, 0, -0.12,  px+0.07, 0.55, 0.12)
    # Asiento
    glColor3f(0.40, 0.25, 0.08)
    box(-0.70, 0.50, -0.18,  0.70, 0.60, 0.18)
    # Respaldo
    box(-0.70, 0.60, -0.18,  0.70, 1.00, -0.10)
    glPopMatrix()


#  POSTE DE LUZ CON CABLES Y BANDERINES

def draw_poste(x, z):
    glPushMatrix()
    glTranslatef(x, 0, z)
    # Poste gris
    glColor3f(0.55, 0.55, 0.55)
    glPushMatrix(); glRotatef(-90,1,0,0)
    cylinder_gl(0.07, 6.5, 8)
    glPopMatrix()
    # Brazo horizontal
    glColor3f(0.45,0.45,0.45)
    box(-0.06, 6.3, -0.8,  0.06, 6.5, 0.06)
    # Foco amarillo
    glColor3f(1.0, 0.95, 0.3)
    glPushMatrix(); glTranslatef(0, 6.35, -0.75)
    sphere_gl(0.13, 8)
    glPopMatrix()
    glPopMatrix()

def draw_cable_banderines(x0, z0, x1, z1, alto=6.2, num_banderas=18):
    """Cable colgante entre dos postes con banderines de colores"""
    colores = [
        (1,0,0),(0,1,0),(0,0,1),(1,1,0),
        (1,0,1),(0,1,1),(1,0.5,0),(0.5,0,1)
    ]
    # Cable 
    glColor3f(0.3,0.3,0.3)
    glLineWidth(1.5)
    glBegin(GL_LINE_STRIP)
    pasos = 40
    for i in range(pasos+1):
        t = i/pasos
        x = x0 + t*(x1-x0)
        z = z0 + t*(z1-z0)
        # curva catenaria simple
        sag = 0.6 * 4*(t*(1-t))
        y = alto - sag
        glVertex3f(x, y, z)
    glEnd()
    glLineWidth(1.0)

    # Banderines triangulares
    for k in range(num_banderas):
        t = (k+0.5)/num_banderas
        x = x0 + t*(x1-x0)
        z = z0 + t*(z1-z0)
        sag = 0.6 * 4*(t*(1-t))
        y = alto - sag
        col = colores[k % len(colores)]
        glColor3f(*col)
        glBegin(GL_TRIANGLES)
        glVertex3f(x-0.12, y,       z)
        glVertex3f(x+0.12, y,       z)
        glVertex3f(x,       y-0.30, z)
        glEnd()


#  CARRO TÍPICO

def draw_car(x, z, rot=0, color=(0.7,0.1,0.1)):
    glPushMatrix()
    glTranslatef(x, 0, z)
    glRotatef(rot, 0,1,0)
    # Carrocería baja
    glColor3f(*color)
    box(-0.9, 0.28, -2.0,  0.9, 0.80, 2.0)
    # Cabina
    glColor3f(color[0]*0.85, color[1]*0.85, color[2]*0.85)
    box(-0.80, 0.80, -1.0,  0.80, 1.35, 1.0)
    # Ventanas (vidrio)
    glColor3f(0.5, 0.7, 0.9)
    box(-0.78, 0.85, -0.95,  0.78, 1.28, -0.93)
    box(-0.78, 0.85,  0.93,  0.78, 1.28,  0.95)
    # Parabrisas
    box(-0.78, 0.85, 0.92,  0.78, 1.28, 0.93)
    # Ruedas
    glColor3f(0.1,0.1,0.1)
    for wx in [-0.9, 0.9]:
        for wz in [-1.3, 1.3]:
            glPushMatrix()
            glTranslatef(wx, 0.28, wz)
            glRotatef(90,0,1,0)
            q = gluNewQuadric()
            gluCylinder(q, 0.28, 0.28, 0.18, 14, 1)
            glPopMatrix()
    # Faro
    glColor3f(1,1,0.7)
    box(-0.7, 0.45,  1.98,  -0.2, 0.65, 2.01)
    box( 0.2, 0.45,  1.98,   0.7, 0.65, 2.01)
    glPopMatrix()

#  PERSONA 
def draw_person(x, z, rot=0, shirt=(0.6,0.1,0.1), skin=(0.85,0.65,0.45)):
    glPushMatrix()
    glTranslatef(x, 0, z)
    glRotatef(rot, 0,1,0)
    # Piernas
    glColor3f(0.15,0.15,0.4)
    box(-0.14, 0,    -0.08,  0.0,  0.72, 0.08)
    box( 0.0,  0,    -0.08,  0.14, 0.72, 0.08)
    # Cuerpo
    glColor3f(*shirt)
    box(-0.18, 0.72, -0.10,  0.18, 1.30, 0.10)
    # Cabeza
    glColor3f(*skin)
    glPushMatrix(); glTranslatef(0, 1.45, 0)
    sphere_gl(0.18, 8)
    glPopMatrix()
    # Sombrero (típico michoacano)
    glColor3f(0.55, 0.35, 0.05)
    glPushMatrix(); glTranslatef(0, 1.58, 0)
    glRotatef(-90,1,0,0)
    cylinder_gl(0.22, 0.18, 12)
    glPopMatrix()
    glColor3f(0.55, 0.35, 0.05)
    glPushMatrix(); glTranslatef(0, 1.63, 0)
    glRotatef(-90,1,0,0)
    cylinder_gl(0.35, 0.04, 12)
    glPopMatrix()
    glPopMatrix()

#casa 
def draw_michoacan_house(x, z, rot=0, w=3.0, d=2.2, wall_col=(0.95,0.93,0.88),
                          roof_col=(0.72,0.20,0.10), door_col=(0.28,0.16,0.06),
                          zocalo_col=(0.40,0.10,0.02), has_second=False):
    glPushMatrix()
    glTranslatef(x, 0, z)
    glRotatef(rot, 0,1,0)
    hw = w/2; hd = d/2; H = 2.6
    H2 = H + 2.4 if has_second else H

    # ── Paredes ──
    glColor3f(*wall_col)
    # Frente
    glBegin(GL_QUADS)
    quad(-hw,0,hd,  hw,0,hd,  hw,H2,hd, -hw,H2,hd)
    # Atrás
    quad( hw,0,-hd, -hw,0,-hd, -hw,H2,-hd,  hw,H2,-hd)
    # Lado izq
    quad(-hw,0,-hd, -hw,0,hd, -hw,H2,hd, -hw,H2,-hd)
    # Lado der
    quad( hw,0,hd,   hw,0,-hd,  hw,H2,-hd,  hw,H2,hd)
    glEnd()

    # ── Zócalo (franja café baja) ──
    glColor3f(*zocalo_col)
    box(-hw-0.01, 0, hd,  hw+0.01, 0.55, hd+0.02)

    # ── Puerta ──
    glColor3f(*door_col)
    box(-0.35, 0, hd+0.01,  0.35, 1.40, hd+0.03)
    # Marco de puerta
    glColor3f(0.9,0.85,0.7)
    box(-0.38, 0, hd+0.02,  -0.32, 1.44, hd+0.04)
    box( 0.32, 0, hd+0.02,   0.38, 1.44, hd+0.04)
    box(-0.38, 1.40, hd+0.02,  0.38, 1.46, hd+0.04)

    # ── Ventana izquierda ──
    glColor3f(0.55,0.73,0.88)
    box(-hw+0.5, 1.0, hd+0.01,  -hw+1.1, 1.7, hd+0.03)
    # Marco
    glColor3f(0.88,0.82,0.65)
    box(-hw+0.46, 0.96, hd+0.02,  -hw+1.14, 1.0, hd+0.04)
    box(-hw+0.46, 1.70, hd+0.02,  -hw+1.14, 1.74, hd+0.04)
    box(-hw+0.46, 0.96, hd+0.02,  -hw+0.50, 1.74, hd+0.04)
    box(-hw+1.10, 0.96, hd+0.02,  -hw+1.14, 1.74, hd+0.04)

    # ── Ventana derecha ──
    glColor3f(0.55,0.73,0.88)
    box( hw-1.1, 1.0, hd+0.01,   hw-0.5, 1.7, hd+0.03)
    glColor3f(0.88,0.82,0.65)
    box( hw-1.14, 0.96, hd+0.02,  hw-0.46, 1.0, hd+0.04)
    box( hw-1.14, 1.70, hd+0.02,  hw-0.46, 1.74, hd+0.04)
    box( hw-1.14, 0.96, hd+0.02,  hw-1.10, 1.74, hd+0.04)
    box( hw-0.50, 0.96, hd+0.02,  hw-0.46, 1.74, hd+0.04)

    # ── Techo de teja (inclinado 2 aguas) ──
    def tejado(base_h, peak_x=0, peak_z=0, ext=0.25):
        ph = base_h + 1.3
        glColor3f(*roof_col)
        glBegin(GL_TRIANGLES)
        # Frente
        glVertex3f(-hw-ext, base_h, hd+ext)
        glVertex3f( hw+ext, base_h, hd+ext)
        glVertex3f( peak_x, ph,     peak_z)
        # Atrás
        glVertex3f( hw+ext, base_h,-hd-ext)
        glVertex3f(-hw-ext, base_h,-hd-ext)
        glVertex3f( peak_x, ph,     peak_z)
        glEnd()
        glBegin(GL_QUADS)
        # Lado izquierdo
        quad(-hw-ext, base_h, hd+ext,  peak_x, ph, peak_z,
             peak_x,  ph, peak_z,       -hw-ext, base_h,-hd-ext)
        # Lado derecho
        quad( hw+ext, base_h,-hd-ext,  peak_x, ph, peak_z,
             peak_x,  ph, peak_z,        hw+ext, base_h, hd+ext)
        glEnd()
        # Canalón (franja oscura en el borde del techo)
        glColor3f(0.50,0.12,0.05)
        box(-hw-ext, base_h-0.08, hd+ext,  hw+ext, base_h, hd+ext+0.08)

    tejado(H2)
    if has_second:
        # Línea entre pisos
        glColor3f(*zocalo_col)
        box(-hw-0.01, H-0.06, -hd,  hw+0.01, H+0.06, hd+0.01)

    # ── Maceta en ventana ──
    glColor3f(0.6,0.2,0.05)
    box(-hw+0.75, 1.0, hd+0.03,  -hw+1.0, 1.08, hd+0.15)
    glColor3f(0.15,0.55,0.1)
    glPushMatrix(); glTranslatef(-hw+0.87, 1.15, hd+0.09)
    sphere_gl(0.1, 6)
    glPopMatrix()

    glPopMatrix()


#  IGLESIA / TEMPLO
def draw_iglesia(x, z):
    glPushMatrix()
    glTranslatef(x, 0, z)

    # Nave principal
    glColor3f(0.97, 0.95, 0.88)
    box(-4.5, 0, -6.0,  4.5, 5.5, 6.0)

    # Techo nave
    glColor3f(0.70, 0.18, 0.08)
    glBegin(GL_TRIANGLES)
    glVertex3f(-5.0, 5.5,-6.5); glVertex3f(5.0, 5.5,-6.5); glVertex3f(0, 7.5, 0)
    glVertex3f( 5.0, 5.5, 6.5); glVertex3f(-5.0,5.5, 6.5); glVertex3f(0, 7.5, 0)
    glEnd()
    glBegin(GL_QUADS)
    quad(-5.0,5.5,6.5,  0,7.5,0,  0,7.5,0,  -5.0,5.5,-6.5)
    quad( 5.0,5.5,-6.5, 0,7.5,0,  0,7.5,0,   5.0,5.5, 6.5)
    glEnd()

    # Torres campanas (2)
    for tx in [-3.5, 3.5]:
        glColor3f(0.97,0.95,0.88)
        box(tx-1.0, 0, 4.5,  tx+1.0, 8.5, 6.5)
        # Arco ventana campanario
        glColor3f(0.35,0.22,0.08)
        box(tx-0.5, 6.0, 6.45,  tx+0.5, 8.0, 6.51)
        # Cúpula
        glColor3f(0.20,0.50,0.20)
        glPushMatrix(); glTranslatef(tx, 8.5, 5.5)
        sphere_gl(1.05, 14)
        glPopMatrix()
        # Cruz
        glColor3f(0.88,0.75,0.1)
        box(tx-0.06, 9.5, 5.44,  tx+0.06, 11.0, 5.56)
        box(tx-0.45, 10.4, 5.44,  tx+0.45, 10.6, 5.56)

    # Portada (fachada decorativa)
    glColor3f(0.93,0.90,0.80)
    box(-4.6, 0, 5.95,  4.6, 5.6, 6.2)
    # Puerta arco
    glColor3f(0.28,0.16,0.06)
    box(-1.0, 0, 6.18,  1.0, 2.8, 6.22)
    # Rosetón (círculo naranja)
    glColor3f(1.0, 0.6, 0.1)
    glPushMatrix(); glTranslatef(0, 4.2, 6.22)
    glRotatef(90,0,1,0)
    q = gluNewQuadric()
    gluCylinder(q, 0.65, 0.65, 0.05, 20, 1)
    glPopMatrix()

    # Escalones
    glColor3f(0.70,0.68,0.65)
    for s in range(4):
        ext = s*0.35
        box(-4.5-ext, s*0.15, 6.2,  4.5+ext, (s+1)*0.15, 6.2+ext+0.35)

    glPopMatrix()


#  KIOSCO CENTRAL

def draw_kiosk(x, z):
    glPushMatrix()
    glTranslatef(x, 0, z)

    # Plataforma octagonal
    glColor3f(0.72, 0.70, 0.65)
    sides = 8
    r_plat = 3.2
    glBegin(GL_POLYGON)
    for i in range(sides):
        a = math.radians(i*360/sides + 22.5)
        glVertex3f(r_plat*math.cos(a), 0.30, r_plat*math.sin(a))
    glEnd()
    # Borde de la plataforma
    glColor3f(0.55,0.53,0.50)
    glBegin(GL_QUAD_STRIP)
    for i in range(sides+1):
        a = math.radians(i*360/sides + 22.5)
        glVertex3f(r_plat*math.cos(a), 0.0,  r_plat*math.sin(a))
        glVertex3f(r_plat*math.cos(a), 0.30, r_plat*math.sin(a))
    glEnd()

    # Escalones
    for s in range(2):
        re = r_plat + 0.5 + s*0.5
        glColor3f(0.65-s*0.05, 0.63-s*0.05, 0.60-s*0.05)
        glBegin(GL_QUAD_STRIP)
        for i in range(sides+1):
            a = math.radians(i*360/sides + 22.5)
            glVertex3f(re*math.cos(a), -s*0.15,     re*math.sin(a))
            glVertex3f(re*math.cos(a), -s*0.15+0.10, re*math.sin(a))
        glEnd()

    # Pasto dentro del kiosco
    glColor3f(0.13, 0.48, 0.13)
    glBegin(GL_POLYGON)
    for i in range(sides):
        a = math.radians(i*360/sides + 22.5)
        glVertex3f((r_plat-0.35)*math.cos(a), 0.31, (r_plat-0.35)*math.sin(a))
    glEnd()

    # Columnas (8)
    r_col = 2.5
    glColor3f(0.92,0.90,0.85)
    for i in range(8):
        a = math.radians(i*45 + 22.5)
        cx, cz = r_col*math.cos(a), r_col*math.sin(a)
        glPushMatrix()
        glTranslatef(cx, 0.30, cz)
        glRotatef(-90,1,0,0)
        cylinder_gl(0.14, 3.8, 10)
        glPopMatrix()
        # Base de columna
        glColor3f(0.80,0.78,0.72)
        box(cx-0.18, 0.30, cz-0.18,  cx+0.18, 0.48, cz+0.18)
        # Capitel
        box(cx-0.20, 4.08, cz-0.20,  cx+0.20, 4.20, cz+0.20)
        glColor3f(0.92,0.90,0.85)

    # Techo octagonal en dos niveles
    def techo_oct(r_base, r_peak, y_base, y_peak, col):
        glColor3f(*col)
        for i in range(sides):
            a0 = math.radians(i*360/sides + 22.5)
            a1 = math.radians((i+1)*360/sides + 22.5)
            glBegin(GL_TRIANGLES)
            glVertex3f(r_base*math.cos(a0), y_base, r_base*math.sin(a0))
            glVertex3f(r_base*math.cos(a1), y_base, r_base*math.sin(a1))
            glVertex3f(0, y_peak, 0)
            glEnd()

    techo_oct(2.75, 0, 4.20, 5.85, (0.72,0.18,0.08))
    techo_oct(1.20, 0, 5.85, 7.00, (0.68,0.15,0.06))
    # Remate/veleta
    glColor3f(0.80,0.65,0.1)
    box(-0.06, 7.0, -0.06,  0.06, 7.9, 0.06)
    glPushMatrix(); glTranslatef(0, 7.95, 0)
    sphere_gl(0.12, 8)
    glPopMatrix()

    # Bancas dentro del kiosco
    for i in range(8):
        a = math.radians(i*45)
        bx, bz = 1.5*math.cos(a), 1.5*math.sin(a)
        draw_bench(bx, bz, rot=math.degrees(a)+90)

    glPopMatrix()


#  ESCUELA

def draw_escuela(x, z, rot=0):
    glPushMatrix()
    glTranslatef(x, 0, z)
    glRotatef(rot, 0,1,0)

    # Cuerpo
    glColor3f(0.95, 0.90, 0.72)
    box(-8.0, 0, -5.0,  8.0, 4.5, 5.0)

    # Zócalo
    glColor3f(0.28, 0.55, 0.28)
    box(-8.05, 0, 4.99,  8.05, 0.70, 5.08)

    # Techo plano con pretil
    glColor3f(0.90, 0.85, 0.68)
    box(-8.2, 4.5, -5.2,  8.2, 4.7, 5.2)
    # Pretil
    glColor3f(0.85,0.80,0.62)
    box(-8.2, 4.7, -5.2,  8.2, 5.2, -4.9)
    box(-8.2, 4.7,  4.9,  8.2, 5.2,  5.2)
    box(-8.2, 4.7, -5.2, -7.9, 5.2,  5.2)
    box( 7.9, 4.7, -5.2,  8.2, 5.2,  5.2)

    # Columnas entrada
    glColor3f(0.97,0.95,0.88)
    for cx in [-2.0, 0.0, 2.0]:
        glPushMatrix(); glTranslatef(cx, 0, 5.01)
        glRotatef(-90,1,0,0)
        cylinder_gl(0.22, 4.5, 10)
        glPopMatrix()

    # Ventanas (6)
    for wx in [-6.0, -3.5, 3.5, 6.0]:
        for wy in [1.0]:
            glColor3f(0.55,0.73,0.88)
            box(wx-0.55, wy, 5.00, wx+0.55, wy+1.5, 5.10)
            glColor3f(0.88,0.82,0.65)
            box(wx-0.60, wy-0.05, 5.06, wx+0.60, wy+1.55, 5.09)

    # Puerta doble
    glColor3f(0.22,0.45,0.22)
    box(-0.70, 0, 5.00,  0.70, 2.5, 5.09)

    # Letrero "ESCUELA PRIMARIA"
    glColor3f(0.15,0.30,0.15)
    box(-3.5, 3.5, 5.05,  3.5, 4.2, 5.12)

    # Mástil bandera
    glColor3f(0.60,0.60,0.60)
    box(-7.8, 0, 4.0,  -7.6, 7.5, 4.2)
    # Bandera México
    glColor3f(0.0, 0.55, 0.27)
    box(-7.6, 6.8, 4.1,  -6.0, 7.5, 4.15)
    glColor3f(0.95,0.95,0.95)
    box(-6.0, 6.8, 4.1,  -4.4, 7.5, 4.15)
    glColor3f(0.80, 0.0, 0.0)
    box(-4.4, 6.8, 4.1,  -2.8, 7.5, 4.15)

    glPopMatrix()


#  FUENTE CENTRAL DE PLAZA
def draw_fuente(x, z):
    glPushMatrix()
    glTranslatef(x, 0, z)
    # Taza grande
    glColor3f(0.72,0.70,0.65)
    q = gluNewQuadric()
    glPushMatrix(); glRotatef(-90,1,0,0)
    gluCylinder(q, 1.8, 1.5, 0.5, 20, 1)
    glPopMatrix()
    # Agua
    glColor3f(0.30, 0.55, 0.80)
    glBegin(GL_POLYGON)
    for i in range(30):
        a = math.radians(i*12)
        glVertex3f(1.55*math.cos(a), 0.48, 1.55*math.sin(a))
    glEnd()
    # Columna central
    glColor3f(0.78,0.76,0.70)
    glPushMatrix(); glTranslatef(0, 0.5, 0); glRotatef(-90,1,0,0)
    gluCylinder(q, 0.18, 0.18, 1.8, 12, 1)
    glPopMatrix()
    # Taza pequeña
    glPushMatrix(); glTranslatef(0, 2.3, 0); glRotatef(-90,1,0,0)
    gluCylinder(q, 0.75, 0.60, 0.35, 16, 1)
    glPopMatrix()
    glPopMatrix()

#  CAMINO DE ADOQUÍN

def draw_adoquin_path(x0, z0, x1, z1, w=2.5):
    """Camino de adoquín (gris moteado)"""
    dx, dz = x1-x0, z1-z0
    length = math.sqrt(dx*dx + dz*dz)
    ang = math.degrees(math.atan2(dx, dz))
    glPushMatrix()
    glTranslatef(x0, 0.005, z0)
    glRotatef(-ang, 0, 1, 0)
    # Base del camino
    glColor3f(0.50, 0.48, 0.45)
    box(-w/2, 0, 0,  w/2, 0.02, length)
    # Adoquines individuales (filas)
    num_r = int(length / 0.55)
    num_c = int(w / 0.55)
    for r in range(num_r):
        for c in range(num_c):
            gris = 0.44 + random.uniform(-0.04, 0.04)
            glColor3f(gris, gris-0.01, gris-0.02)
            rx = -w/2 + c*0.55 + 0.04
            rz = r*0.55 + 0.04
            box(rx, 0.005, rz,  rx+0.50, 0.025, rz+0.50)
    glPopMatrix()


#  JARDIN / PASTO  con flores

def draw_jardin(x0, z0, x1, z1):
    glColor3f(0.13, 0.48, 0.14)
    glBegin(GL_QUADS)
    glVertex3f(x0,0.01,z0); glVertex3f(x1,0.01,z0)
    glVertex3f(x1,0.01,z1); glVertex3f(x0,0.01,z1)
    glEnd()
    # Flores aleatorias
    colores_flores = [(1,0,0),(1,0.8,0),(1,0,0.8),(1,1,0),(1,0.4,0)]
    rng = random.Random(77)
    for _ in range(60):
        fx = rng.uniform(x0+0.3, x1-0.3)
        fz = rng.uniform(z0+0.3, z1-0.3)
        col = colores_flores[rng.randint(0,4)]
        glColor3f(*col)
        glPushMatrix(); glTranslatef(fx, 0.05, fz)
        sphere_gl(0.09, 6)
        glPopMatrix()
        # Tallo
        glColor3f(0.1,0.5,0.1)
        glPushMatrix(); glTranslatef(fx, 0.0, fz)
        glRotatef(-90,1,0,0)
        cylinder_gl(0.02, 0.10, 5)
        glPopMatrix()

#  BANCO EXTERIOR / NEGOCIOS

def draw_negocio(x, z, rot=0, color=(0.88,0.72,0.35), letrero_col=(0.6,0.05,0.05)):
    """Pequeño negocio / tienda michoacana"""
    glPushMatrix()
    glTranslatef(x, 0, z)
    glRotatef(rot, 0,1,0)
    # Cuerpo
    glColor3f(*color)
    box(-2.2, 0, -1.5,  2.2, 3.2, 1.5)
    # Zócalo
    glColor3f(0.35,0.08,0.0)
    box(-2.21, 0, 1.49,  2.21, 0.5, 1.58)
    # Toldo
    glColor3f(*letrero_col)
    box(-2.4, 2.5, 1.4,  2.4, 2.7, 2.4)
    # Vitrina
    glColor3f(0.65,0.82,0.92)
    box(-1.5, 0.5, 1.49,  1.5, 2.3, 1.56)
    # Puerta
    glColor3f(0.28,0.16,0.06)
    box(-0.4, 0, 1.49,  0.4, 2.2, 1.57)
    glPopMatrix()


#  MONTAÑAS DE FONDO

def draw_mountains():
    peaks = [
        (-80, 35, -95), (0, 45, -100), (80, 30, -90),
        (-120, 25, -85), (120, 28, -88),
        (-50, 20, -80), (50, 22, -80),
    ]
    for (mx, mh, mz) in peaks:
        glColor3f(0.28, 0.38, 0.22)
        glBegin(GL_TRIANGLES)
        glVertex3f(mx-30, 0,   mz+15)
        glVertex3f(mx+30, 0,   mz+15)
        glVertex3f(mx,    mh,  mz)
        glEnd()
        # Nieve
        glColor3f(0.92, 0.93, 0.95)
        glBegin(GL_TRIANGLES)
        glVertex3f(mx-8, mh*0.72, mz+4)
        glVertex3f(mx+8, mh*0.72, mz+4)
        glVertex3f(mx,   mh,      mz)
        glEnd()


#  CIELO CON NUBES

def draw_clouds():
    cloud_data = [
        (-20, 22, -60), (10, 25, -70), (35, 20, -55),
        (-40, 24, -75), (55, 23, -65),
    ]
    glColor3f(0.97, 0.97, 0.97)
    for (cx,cy,cz) in cloud_data:
        for dx, dz, r in [(0,0,2.2),(-2,0,1.5),(2,0,1.5),(0,0.8,1.3)]:
            glPushMatrix()
            glTranslatef(cx+dx, cy, cz+dz)
            sphere_gl(r, 10)
            glPopMatrix()

#  ESCENA PRINCIPAL

def draw_pueblo():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    rad = math.radians(cam_yaw)
    lx = cam_x - math.sin(rad)*10
    lz = cam_z - math.cos(rad)*10
    gluLookAt(cam_x, cam_y, cam_z,  lx, cam_y, lz,  0,1,0)

    # ── Cielo / Nubes / Montañas ──────────────
    draw_clouds()
    draw_mountains()

    # ── Suelo general empedrado ───────────────
    glColor3f(0.38, 0.36, 0.33)
    glBegin(GL_QUADS)
    glVertex3f(-100, 0, 100); glVertex3f(100, 0, 100)
    glVertex3f(100, 0,-100); glVertex3f(-100, 0,-100)
    glEnd()

    # ── Jardín de la plaza ────────────────────
    draw_jardin(-11, -11, 11, 11)

    # ── Caminos adoquinados (4 ejes desde plaza)
    draw_adoquin_path( 0, 11,  0,  50, w=3.0)   # Norte
    draw_adoquin_path( 0,-11,  0, -50, w=3.0)   # Sur
    draw_adoquin_path( 11, 0, 50,   0, w=3.0)   # Este
    draw_adoquin_path(-11, 0,-50,   0, w=3.0)   # Oeste
    # Caminos diagonales secundarios
    draw_adoquin_path( 11, 11, 45,  45, w=2.2)
    draw_adoquin_path(-11, 11,-45,  45, w=2.2)
    draw_adoquin_path( 11,-11, 45, -45, w=2.2)
    draw_adoquin_path(-11,-11,-45, -45, w=2.2)

    # ── Kiosco central 
    draw_kiosk(0, 0)

    # ── Fuente 
    draw_fuente(0, -7.5)
    draw_fuente(0,  7.5)

    # ── Iglesia 
    draw_iglesia(0, -42)

    # ── Escuela
    draw_escuela(35, 40, rot=180)

    casa_norte = [(x, 18) for x in range(-28, 32, 14)]
    for (hx, hz) in casa_norte:
        draw_michoacan_house(hx, hz, rot=180, w=3.5, d=2.5,
                              has_second=(hx % 14 == 0))
    # Sur
    casa_sur = [(x, -18) for x in range(-28, 32, 14)]
    for (hx, hz) in casa_sur:
        draw_michoacan_house(hx, hz, rot=0, w=3.5, d=2.5,
                              wall_col=(0.92,0.90,0.82),
                              roof_col=(0.68,0.18,0.06))
    # Este
    casa_este = [(20, z) for z in range(-26, 20, 13)]
    for (hx, hz) in casa_este:
        draw_michoacan_house(hx, hz, rot=270, w=3.0, d=2.2,
                              wall_col=(0.93,0.88,0.78),
                              roof_col=(0.65,0.15,0.08))
    # Oeste
    casa_oeste = [(-20, z) for z in range(-26, 20, 13)]
    for (hx, hz) in casa_oeste:
        draw_michoacan_house(hx, hz, rot=90, w=3.0, d=2.2,
                              wall_col=(0.88,0.85,0.75),
                              roof_col=(0.70,0.20,0.10))

    # Casas en anillo exterior
    for i, hx in enumerate(range(-45, 50, 15)):
        draw_michoacan_house(hx, 35, rot=180, w=4.0, d=2.8,
                              has_second=(i%2==0))
        draw_michoacan_house(hx,-35, rot=0,   w=4.0, d=2.8)
    for i, hz in enumerate(range(-32, 36, 14)):
        draw_michoacan_house( 40, hz, rot=270, w=3.5, d=2.4)
        draw_michoacan_house(-40, hz, rot=90,  w=3.5, d=2.4,
                              wall_col=(0.90,0.87,0.76))

    # ── Negocios 
    draw_negocio(15,  16, rot=180, color=(0.85,0.70,0.30), letrero_col=(0.1,0.35,0.1))
    draw_negocio(-15, 16, rot=180, color=(0.80,0.65,0.28), letrero_col=(0.6,0.05,0.05))
    draw_negocio( 15,-16, rot=0,   color=(0.78,0.60,0.25), letrero_col=(0.1,0.10,0.55))
    draw_negocio(-15,-16, rot=0,   color=(0.82,0.68,0.28), letrero_col=(0.55,0.35,0.0))

    # ── Árboles 
    arboles_plaza = [
        (-8, -8), ( 8, -8), (-8, 8), (8, 8),
        (-3, -9), ( 3, -9), (-3,  9),(3,  9),
    ]
    for i,(ax,az) in enumerate(arboles_plaza):
        draw_tree(ax, az, h=3.0+i%2, kind=i%2)

    # Árboles en calles
    for zz in range(-45, 46, 8):
        draw_tree( 14, zz, h=4.0, kind=0)
        draw_tree(-14, zz, h=3.5+((zz+45)%3)*0.5, kind=1)
    for xx in range(-45, 46, 8):
        draw_tree(xx,  13, h=4.0, kind=1)
        draw_tree(xx, -13, h=3.5, kind=0)
    # Árboles dispersos
    for (ax,az,k) in [(-30,25,0),(30,25,1),(-30,-25,1),(30,-25,0),
                       (-50,0,0),(50,0,1),(0,50,0),(0,-50,1),
                       (22,42,0),(-22,42,1),(22,-42,0),(-22,-42,1)]:
        draw_tree(ax, az, h=4.5, kind=k)

    # ── Postes y banderines 
    postes_norte = [(-12, 14), (-6, 14), (0, 14), (6, 14), (12, 14)]
    postes_sur   = [(-12,-14), (-6,-14), (0,-14), (6,-14), (12,-14)]
    postes_este  = [(13,-12),  (13,-6),  (13, 0), (13, 6), (13,12)]
    postes_oeste = [(-13,-12),(-13,-6), (-13, 0),(-13, 6),(-13,12)]

    for (px,pz) in postes_norte + postes_sur + postes_este + postes_oeste:
        draw_poste(px, pz)

    # Cables con banderines entre postes
    for i in range(len(postes_norte)-1):
        p0, p1 = postes_norte[i], postes_norte[i+1]
        draw_cable_banderines(p0[0], p0[1], p1[0], p1[1])
    for i in range(len(postes_sur)-1):
        p0, p1 = postes_sur[i], postes_sur[i+1]
        draw_cable_banderines(p0[0], p0[1], p1[0], p1[1])
    for i in range(len(postes_este)-1):
        p0, p1 = postes_este[i], postes_este[i+1]
        draw_cable_banderines(p0[0], p0[1], p1[0], p1[1], num_banderas=12)
    for i in range(len(postes_oeste)-1):
        p0, p1 = postes_oeste[i], postes_oeste[i+1]
        draw_cable_banderines(p0[0], p0[1], p1[0], p1[1], num_banderas=12)

    # Cables fachada-a-fachada 
    for zz in [-10, -5, 0, 5, 10]:
        draw_cable_banderines(-13, zz, 13, zz, num_banderas=14)

    # ── Coches 
    draw_car( 5,  30, rot=0,   color=(0.15,0.25,0.65))
    draw_car(-5,  30, rot=180, color=(0.7, 0.1, 0.1))
    draw_car(16,  5,  rot=90,  color=(0.1, 0.5, 0.15))
    draw_car(16, -8,  rot=270, color=(0.7, 0.6, 0.1))
    draw_car( 0, -32, rot=0,   color=(0.5, 0.1, 0.5))
    draw_car(-18, 10, rot=90,  color=(0.15,0.15,0.15))
    draw_car( 3,  50, rot=180, color=(0.85,0.40,0.10))

    # ── Gente 
    personas = [
        ( 5,  4, 45,  (0.80,0.10,0.10)),
        (-4,  6, 90,  (0.10,0.30,0.65)),
        ( 0,  5, 0,   (0.50,0.10,0.10)),
        (-6, -4, 200, (0.20,0.55,0.20)),
        ( 8, -6, 135, (0.65,0.40,0.10)),
        (-3, -8, 270, (0.85,0.55,0.10)),
        (14,  9, 180, (0.60,0.10,0.35)),
        (-14, 8, 10,  (0.10,0.10,0.60)),
        ( 2, 15, 220, (0.70,0.20,0.05)),
        (-2,-15, 60,  (0.15,0.45,0.15)),
        ( 7, 18, 0,   (0.80,0.50,0.10)),
        (-7,-18, 180, (0.50,0.20,0.60)),
        (22,  3, 90,  (0.90,0.20,0.10)),
        (-22, 3, 270, (0.20,0.60,0.20)),
    ]
    for (px, pz, prot, shirt) in personas:
        draw_person(px, pz, rot=prot, shirt=shirt)

    # ── Bancas de la plaza ────────
    for ang_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
        a = math.radians(ang_deg)
        bx, bz = 9.5*math.cos(a), 9.5*math.sin(a)
        draw_bench(bx, bz, rot=ang_deg+90)

    glfw.swap_buffers(window)
def main():
    global window

    if not glfw.init():
        return

    window = glfw.create_window(
        1000,
        700,
        "Control con mano",
        None,
        None
    )

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    init()

    while not glfw.window_should_close(window):
        hand_control()      # detectar mano
        draw_pueblo()       # dibujar escena
        glfw.poll_events()

    cap.release()
    cv2.destroyAllWindows()
    glfw.terminate()


if __name__ == "__main__":
    main()
