import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"

import math, time, threading, random
import glfw, cv2
import numpy as np
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GLU import *

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode
#CORRIJAN LA UBICACION POR QUE ESTA ES LA MIA 
MODEL_PATH = r'C:\Users\tigre\OneDrive\Documentos\graficacion\hand_landmarker.task'

gesture_lock     = threading.Lock()
current_gesture  = "NONE"
cam_frame_lock   = threading.Lock()
cam_frame_latest = None

cam_x, cam_y, cam_z = 0.0, 1.75, 28.0
cam_yaw   =  0.0
cam_pitch = -8.0
VEL_MOVE  =  0.22
VEL_ROT   =  1.8

random.seed(42)
t_start = time.time()
def T(): return time.time() - t_start

PERSONAS_DATA = [
    (  5,   4,  45, 0.04, 4.0, (0.80,0.10,0.10)),
    ( -4,   6, 120, 0.03, 5.0, (0.10,0.30,0.65)),
    (  2,   5,   0, 0.05, 3.5, (0.50,0.10,0.10)),
    ( -6,  -4, 200, 0.03, 4.5, (0.20,0.55,0.20)),
    (  8,  -6, 135, 0.04, 3.0, (0.65,0.40,0.10)),
    ( -3,  -8, 270, 0.05, 5.5, (0.85,0.55,0.10)),
    ( 14,   9, 180, 0.03, 6.0, (0.60,0.10,0.35)),
    (-14,   8,  10, 0.04, 5.0, (0.10,0.10,0.60)),
    (  2,  15, 220, 0.05, 4.0, (0.70,0.20,0.05)),
    ( -2, -15,  60, 0.04, 4.5, (0.15,0.45,0.15)),
    (  7,  18,   0, 0.03, 3.5, (0.80,0.50,0.10)),
    ( -7, -18, 180, 0.05, 4.0, (0.50,0.20,0.60)),
    ( 22,   3,  90, 0.04, 5.0, (0.90,0.20,0.10)),
    (-22,   3, 270, 0.03, 4.5, (0.20,0.60,0.20)),
]

BURROS_DATA = [
    ( -8, 20,  90, 0.025, 8.0),
    ( 25,-18, 200, 0.020, 6.0),
]

def finger_extended(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def classify_gesture(hand_landmarks_list):
    lm = hand_landmarks_list # Lista de puntos normalizados de MediaPipe
    
    # Estados de los dedos
    thumb_up   = lm[4].y < lm[3].y < lm[2].y
    thumb_down = lm[4].y > lm[3].y > lm[2].y
    index      = finger_extended(lm, 8, 6)
    middle     = finger_extended(lm, 12, 10)
    ring       = finger_extended(lm, 16, 14)
    pinky      = finger_extended(lm, 20, 18)
    
    # 1. PALMA ABIERTA -> AVANZAR (Todos los dedos extendidos)
    if index and middle and ring and pinky:
        return "FORWARD"
    
    # 2. PUÑO CERRADO -> RETROCEDER (Ningún dedo extendido, excepto quizás el pulgar neutral)
    if not index and not middle and not ring and not pinky and not thumb_up:
        return "BACK"

    # 3. SÍMBOLO DE PAZ -> GIRAR IZQUIERDA (Índice y Medio)
    if index and middle and not ring and not pinky:
        return "TURN_L"

    # 4. ÍNDICE -> GIRAR DERECHA (Solo el índice)
    if index and not middle and not ring and not pinky:
        return "TURN_R"

    # 5. PULGAR ARRIBA -> SUBIR
    #if thumb_up and not index and not middle and not ring and not pinky:
    #    return "UP"

    # 6. PULGAR ABAJO -> BAJAR
    #if thumb_down and not index and not middle and not ring and not pinky:
     #   return "DOWN"

    return "NONE"
labels = {
    "FORWARD": "Avanzar",
    "BACK":    "Retroceder",
    "TURN_R":  "Girar IZQUIERDA",
    "TURN_L":  "Girar DERECHA",
    "UP":      "Subir",
    "DOWN":    "Bajar",
    "NONE":    "Sin gesto",
}
colors_hud = {
    "FORWARD": (0,230,0),
    "BACK":    (0,0,230),
    "TURN_R":  (230,180,0),
    "TURN_L":  (230,100,0),
    "UP":      (180,0,230),
    "DOWN":    (100,100,230),
    "NONE":    (180,180,180),
}


def on_hand_result(result, output_image, timestamp_ms):
    global current_gesture
    gesture = "NONE"
    if result and result.hand_landmarks:
        gesture = classify_gesture(result.hand_landmarks[0])
    with gesture_lock:
        current_gesture = gesture

def vision_thread():
    global cam_frame_latest

    if not os.path.exists(MODEL_PATH):
        print("Modelo no encontrado: " + MODEL_PATH)
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_hand_result,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Inicializar marcas de tiempo incrementales obligatorias en LIVE_STREAM
    base_timestamp = int(time.time() * 1000)

    with HandLandmarker.create_from_options(options) as detector:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)

            if frame_idx % 2 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                
                
                current_timestamp = base_timestamp + int(time.time() * 1000) - base_timestamp
                try:
                    detector.detect_async(mp_image, current_timestamp)
                except Exception as e:
                    pass

            frame_idx += 1

            with gesture_lock:
                g = current_gesture
            col = colors_hud.get(g, (180,180,180))
            cv2.putText(frame, labels.get(g, ""), (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
            cv2.rectangle(frame, (0,0), (frame.shape[1]-1, frame.shape[0]-1), col, 2)

            with cam_frame_lock:
                cam_frame_latest = frame.copy()

            time.sleep(0.02) # Sincronización más fluida de captura

    cap.release()

keys_held = set()
def key_callback(window, key, scancode, action, mods):
    if action == glfw.PRESS:   keys_held.add(key)
    if action == glfw.RELEASE: keys_held.discard(key)
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

def apply_camera_movement():
    global cam_x, cam_y, cam_z, cam_yaw, cam_pitch
    with gesture_lock:
        g = current_gesture
    yr = math.radians(cam_yaw)
    fx, fz = -math.sin(yr), -math.cos(yr)
    
    if g == "FORWARD": cam_x += fx*VEL_MOVE; cam_z += fz*VEL_MOVE
    elif g == "BACK":  cam_x -= fx*VEL_MOVE; cam_z -= fz*VEL_MOVE
    elif g == "TURN_R": cam_yaw += VEL_ROT
    elif g == "TURN_L": cam_yaw -= VEL_ROT
    elif g == "UP":   cam_y += VEL_MOVE*0.6
    elif g == "DOWN": cam_y -= VEL_MOVE*0.6
    
    if glfw.KEY_W in keys_held or glfw.KEY_UP    in keys_held: cam_x += fx*VEL_MOVE; cam_z += fz*VEL_MOVE
    if glfw.KEY_S in keys_held or glfw.KEY_DOWN  in keys_held: cam_x -= fx*VEL_MOVE; cam_z -= fz*VEL_MOVE
    if glfw.KEY_A in keys_held or glfw.KEY_LEFT  in keys_held: cam_yaw -= VEL_ROT
    if glfw.KEY_D in keys_held or glfw.KEY_RIGHT in keys_held: cam_yaw += VEL_ROT
    if glfw.KEY_Q in keys_held: cam_y += VEL_MOVE*0.6
    if glfw.KEY_Z in keys_held: cam_y -= VEL_MOVE*0.6
    cam_pitch = max(-89.0, min(89.0, cam_pitch))

def init_gl():
    glClearColor(0.52, 0.73, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(65, 1000/700, 0.1, 300.0)
    glMatrixMode(GL_MODELVIEW)

def quad(x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3):
    glVertex3f(x0,y0,z0); glVertex3f(x1,y1,z1)
    glVertex3f(x2,y2,z2); glVertex3f(x3,y3,z3)

def box(x0,y0,z0,x1,y1,z1):
    glBegin(GL_QUADS)
    quad(x0,y0,z1,x1,y0,z1,x1,y1,z1,x0,y1,z1)
    quad(x1,y0,z0,x0,y0,z0,x0,y1,z0,x1,y1,z0)
    quad(x0,y0,z0,x0,y0,z1,x0,y1,z1,x0,y1,z0)
    quad(x1,y0,z1,x1,y0,z0,x1,y1,z0,x1,y1,z1)
    quad(x0,y1,z1,x1,y1,z1,x1,y1,z0,x0,y1,z0)
    quad(x0,y0,z0,x1,y0,z0,x1,y0,z1,x0,y0,z1)
    glEnd()

def cyl(r,h,sl=12):
    q=gluNewQuadric(); gluCylinder(q,r,r,h,sl,1)

def sph(r,sl=10):
    q=gluNewQuadric(); gluSphere(q,r,sl,sl)

def draw_tree(x,z,h=3.5,cr=0.9,kind=0):
    glPushMatrix(); glTranslatef(x,0,z)
    glColor3f(0.35,0.22,0.08)
    glPushMatrix(); glRotatef(-90,1,0,0); cyl(0.15,h*0.38); glPopMatrix()
    by=h*0.38
    if kind==0:
        glColor3f(0.08,0.38,0.08)
        for i in range(3):
            r=cr*(1-i*0.22); hy=by+i*(h*0.20)
            glPushMatrix(); glTranslatef(0,hy,0); glRotatef(-90,1,0,0); cyl(r,h*0.35,16); glPopMatrix()
    else:
        glColor3f(0.1,0.45,0.1)
        glPushMatrix(); glTranslatef(0,by+cr*0.85,0); sph(cr,14); glPopMatrix()
    glPopMatrix()

def draw_bench(x,z,rot=0):
    glPushMatrix(); glTranslatef(x,0,z); glRotatef(rot,0,1,0)
    glColor3f(0.25,0.15,0.05)
    for px in [-0.55,0.55]: box(px-0.07,0,-0.12,px+0.07,0.55,0.12)
    glColor3f(0.40,0.25,0.08)
    box(-0.70,0.50,-0.18,0.70,0.60,0.18)
    box(-0.70,0.60,-0.18,0.70,1.00,-0.10)
    glPopMatrix()

def draw_poste(x,z):
    glPushMatrix(); glTranslatef(x,0,z)
    glColor3f(0.55,0.55,0.55)
    glPushMatrix(); glRotatef(-90,1,0,0); cyl(0.07,6.5,8); glPopMatrix()
    glColor3f(0.45,0.45,0.45); box(-0.06,6.3,-0.8,0.06,6.5,0.06)
    glColor3f(1.0,0.95,0.3)
    glPushMatrix(); glTranslatef(0,6.35,-0.75); sph(0.13,8); glPopMatrix()
    glPopMatrix()

def draw_cable_banderines(x0,z0,x1,z1,alto=6.2,nb=18):
    cols=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,.5,0),(.5,0,1)]
    glColor3f(0.3,0.3,0.3); glLineWidth(1.5)
    glBegin(GL_LINE_STRIP)
    for i in range(41):
        t=i/40; sag=0.6*4*t*(1-t)
        glVertex3f(x0+t*(x1-x0),alto-sag,z0+t*(z1-z0))
    glEnd(); glLineWidth(1.0)
    for k in range(nb):
        t=(k+.5)/nb; sag=0.6*4*t*(1-t)
        x=x0+t*(x1-x0); z=z0+t*(z1-z0); y=alto-sag
        glColor3f(*cols[k%len(cols)])
        glBegin(GL_TRIANGLES)
        glVertex3f(x-.12,y,z); glVertex3f(x+.12,y,z); glVertex3f(x,y-.30,z)
        glEnd()

def draw_car(x,z,rot=0,color=(0.7,0.1,0.1)):
    glPushMatrix(); glTranslatef(x,0,z); glRotatef(rot,0,1,0)
    glColor3f(*color); box(-0.9,0.28,-2.0,0.9,0.80,2.0)
    c2=(color[0]*.85,color[1]*.85,color[2]*.85)
    glColor3f(*c2); box(-0.80,0.80,-1.0,0.80,1.35,1.0)
    glColor3f(0.5,0.7,0.9)
    box(-0.78,0.85,-0.95,0.78,1.28,-0.93)
    box(-0.78,0.85,.93,0.78,1.28,.95)
    glColor3f(.1,.1,.1)
    for wx in [-0.9,0.9]:
        for wz in [-1.3,1.3]:
            glPushMatrix(); glTranslatef(wx,0.28,wz); glRotatef(90,0,1,0)
            q=gluNewQuadric(); gluCylinder(q,.28,.28,.18,14,1); glPopMatrix()
    glColor3f(1,1,0.7)
    box(-.7,.45,1.98,-.2,.65,2.01); box(.2,.45,1.98,.7,.65,2.01)
    glPopMatrix()

def animated_pos(sx,sz,dir0,spd,radio,t):
    ang=math.radians(dir0)+t*spd
    return sx+radio*math.cos(ang), sz+radio*math.sin(ang), (math.degrees(ang)+90)%360

def draw_person_at(px,pz,face_deg=0,anim_t=0,shirt=(0.6,.1,.1),skin=(0.85,.65,.45)):
    glPushMatrix(); glTranslatef(px,0,pz); glRotatef(face_deg,0,1,0)
    sw=math.sin(anim_t*5.0)*15.0
    glColor3f(0.15,0.15,0.4)
    glPushMatrix(); glTranslatef(-0.07,0.72,0); glRotatef(sw,1,0,0)
    box(-.07,-.72,-.08,.07,0,.08); glPopMatrix()
    glPushMatrix(); glTranslatef(0.07,0.72,0); glRotatef(-sw,1,0,0)
    box(-.07,-.72,-.08,.07,0,.08); glPopMatrix()
    glColor3f(*shirt); box(-.18,.72,-.10,.18,1.30,.10)
    glPushMatrix(); glTranslatef(-.22,1.25,0); glRotatef(-sw,1,0,0)
    box(-.07,-.45,-.07,.07,0,.07); glPopMatrix()
    glPushMatrix(); glTranslatef(.22,1.25,0); glRotatef(sw,1,0,0)
    box(-.07,-.45,-.07,.07,0,.07); glPopMatrix()
    glColor3f(*skin)
    glPushMatrix(); glTranslatef(0,1.45,0); sph(0.18,8); glPopMatrix()
    glColor3f(0.55,0.35,0.05)
    glPushMatrix(); glTranslatef(0,1.58,0); glRotatef(-90,1,0,0); cyl(.22,.18,12); glPopMatrix()
    glPushMatrix(); glTranslatef(0,1.63,0); glRotatef(-90,1,0,0); cyl(.35,.04,12); glPopMatrix()
    glPopMatrix()

def draw_burro_at(px,pz,face_deg=0,anim_t=0):
    glPushMatrix(); glTranslatef(px,0,pz); glRotatef(face_deg,0,1,0)
    sw=math.sin(anim_t*4.0)*12.0
    glColor3f(0.55,0.52,0.48); box(-.45,.55,-.90,.45,1.05,.90)
    glColor3f(0.50,0.47,0.43); box(-.18,.95,.70,.18,1.55,1.00)
    glColor3f(0.50,0.47,0.43)
    glPushMatrix(); glTranslatef(0,1.45,.95)
    box(-.22,-.15,-.10,.22,.32,.45); glPopMatrix()
    glColor3f(0.45,0.40,0.38)
    glPushMatrix(); glTranslatef(.12,1.72,1.05); glRotatef(-15,0,0,1)
    box(-.04,0,-.03,.04,.35,.03); glPopMatrix()
    glPushMatrix(); glTranslatef(-.12,1.72,1.05); glRotatef(15,0,0,1)
    box(-.04,0,-.03,.04,.35,.03); glPopMatrix()
    glColor3f(0.40,0.37,0.33)
    for lx,lz,s in [(-.30,-.55,1),(-.30,.55,-1),(.30,-.55,-1),(.30,.55,1)]:
        glPushMatrix(); glTranslatef(lx,.55,lz); glRotatef(sw*s,1,0,0)
        box(-.08,-.55,-.06,.08,0,.06); glPopMatrix()
    glColor3f(0.30,0.25,0.20)
    glPushMatrix(); glTranslatef(0,.85,-.92); glRotatef(20,1,0,0); cyl(.03,.50,6); glPopMatrix()
    glPopMatrix()

def draw_house(x,z,rot=0,w=3.0,d=2.2,wc=(0.95,.93,.88),rc=(.72,.20,.10),
               dc=(.28,.16,.06),zc=(.40,.10,.02),sec=False):
    glPushMatrix(); glTranslatef(x,0,z); glRotatef(rot,0,1,0)
    hw=w/2; hd=d/2; H=2.6; H2=H+2.4 if sec else H
    glColor3f(*wc)
    glBegin(GL_QUADS)
    quad(-hw,0,hd,hw,0,hd,hw,H2,hd,-hw,H2,hd)
    quad(hw,0,-hd,-hw,0,-hd,-hw,H2,-hd,hw,H2,-hd)
    quad(-hw,0,-hd,-hw,0,hd,-hw,H2,hd,-hw,H2,-hd)
    quad(hw,0,hd,hw,0,-hd,hw,H2,-hd,hw,H2,hd)
    glEnd()
    glColor3f(*zc); box(-hw-.01,0,hd,hw+.01,.55,hd+.02)
    glColor3f(*dc); box(-.35,0,hd+.01,.35,1.40,hd+.03)
    glColor3f(.9,.85,.7)
    box(-.38,0,hd+.02,-.32,1.44,hd+.04)
    box(.32,0,hd+.02,.38,1.44,hd+.04)
    box(-.38,1.40,hd+.02,.38,1.46,hd+.04)
    glColor3f(.55,.73,.88)
    box(-hw+.5,1.0,hd+.01,-hw+1.1,1.7,hd+.03)
    box(hw-1.1,1.0,hd+.01,hw-.5,1.7,hd+.03)
    glColor3f(.88,.82,.65)
    box(-hw+.46,.96,hd+.02,-hw+1.14,1.0,hd+.04)
    box(-hw+.46,1.70,hd+.02,-hw+1.14,1.74,hd+.04)
    box(-hw+.46,.96,hd+.02,-hw+.50,1.74,hd+.04)
    box(-hw+1.10,.96,hd+.02,-hw+1.14,1.74,hd+.04)
    ph=H2+1.3; ext=0.25
    glColor3f(*rc)
    glBegin(GL_TRIANGLES)
    glVertex3f(-hw-ext,H2,hd+ext); glVertex3f(hw+ext,H2,hd+ext); glVertex3f(0,ph,0)
    glVertex3f(hw+ext,H2,-hd-ext); glVertex3f(-hw-ext,H2,-hd-ext); glVertex3f(0,ph,0)
    glEnd()
    glBegin(GL_QUADS)
    quad(-hw-ext,H2,hd+ext,0,ph,0,0,ph,0,-hw-ext,H2,-hd-ext)
    quad(hw+ext,H2,-hd-ext,0,ph,0,0,ph,0,hw+ext,H2,hd+ext)
    glEnd()
    glColor3f(.50,.12,.05); box(-hw-ext,H2-.08,hd+ext,hw+ext,H2,hd+ext+.08)
    glPopMatrix()

def draw_kiosk(x,z):
    glPushMatrix(); glTranslatef(x,0,z)
    S=8; rp=3.2
    glColor3f(.72,.70,.65)
    glBegin(GL_POLYGON)
    for i in range(S):
        a=math.radians(i*360/S+22.5); glVertex3f(rp*math.cos(a),.30,rp*math.sin(a))
    glEnd()
    glColor3f(.55,.53,.50)
    glBegin(GL_QUAD_STRIP)
    for i in range(S+1):
        a=math.radians(i*360/S+22.5)
        glVertex3f(rp*math.cos(a),0,rp*math.sin(a))
        glVertex3f(rp*math.cos(a),.30,rp*math.sin(a))
    glEnd()
    for s in range(2):
        re=rp+.5+s*.5; glColor3f(.65-s*.05,.63-s*.05,.60-s*.05)
        glBegin(GL_QUAD_STRIP)
        for i in range(S+1):
            a=math.radians(i*360/S+22.5)
            glVertex3f(re*math.cos(a),-s*.15,re*math.sin(a))
            glVertex3f(re*math.cos(a),-s*.15+.10,re*math.sin(a))
        glEnd()
    glColor3f(.13,.48,.13)
    glBegin(GL_POLYGON)
    for i in range(S):
        a=math.radians(i*360/S+22.5); r=(rp-.35)
        glVertex3f(r*math.cos(a),.31,r*math.sin(a))
    glEnd()
    rc=2.5; glColor3f(.92,.90,.85)
    for i in range(8):
        a=math.radians(i*45+22.5); cx,cz=rc*math.cos(a),rc*math.sin(a)
        glPushMatrix(); glTranslatef(cx,.30,cz); glRotatef(-90,1,0,0)
        cyl(.14,3.8,10); glPopMatrix()
        glColor3f(.80,.78,.72); box(cx-.18,.30,cz-.18,cx+.18,.48,cz+.18)
        box(cx-.20,4.08,cz-.20,cx+.20,4.20,cz+.20); glColor3f(.92,.90,.85)
    def toct(rb,yb,yp,col):
        glColor3f(*col)
        for i in range(S):
            a0=math.radians(i*360/S+22.5); a1=math.radians((i+1)*360/S+22.5)
            glBegin(GL_TRIANGLES)
            glVertex3f(rb*math.cos(a0),yb,rb*math.sin(a0))
            glVertex3f(rb*math.cos(a1),yb,rb*math.sin(a1))
            glVertex3f(0,yp,0)
            glEnd()
    toct(2.75,4.20,5.85,(.72,.18,.08))
    toct(1.20,5.85,7.00,(.68,.15,.06))
    glColor3f(.80,.65,.1); box(-.06,7.0,-.06,.06,7.9,.06)
    glPushMatrix(); glTranslatef(0,7.95,0); sph(.12,8); glPopMatrix()
    for i in range(8):
        a=math.radians(i*45); draw_bench(1.5*math.cos(a),1.5*math.sin(a),rot=math.degrees(a)+90)
    glPopMatrix()

def draw_iglesia(x,z):
    glPushMatrix(); glTranslatef(x,0,z)
    glColor3f(.97,.95,.88); box(-4.5,0,-6.0,4.5,5.5,6.0)
    glColor3f(.70,.18,.08)
    glBegin(GL_TRIANGLES)
    glVertex3f(-5.0,5.5,-6.5); glVertex3f(5.0,5.5,-6.5); glVertex3f(0,7.5,0)
    glVertex3f(5.0,5.5,6.5);   glVertex3f(-5.0,5.5,6.5); glVertex3f(0,7.5,0)
    glEnd()
    glBegin(GL_QUADS)
    quad(-5.0,5.5,6.5,0,7.5,0,0,7.5,0,-5.0,5.5,-6.5)
    quad(5.0,5.5,-6.5,0,7.5,0,0,7.5,0,5.0,5.5,6.5)
    glEnd()
    for tx in [-3.5,3.5]:
        glColor3f(.97,.95,.88); box(tx-1.0,0,4.5,tx+1.0,8.5,6.5)
        glColor3f(.35,.22,.08); box(tx-.5,6.0,6.45,tx+.5,8.0,6.51)
        glColor3f(.20,.50,.20)
        glPushMatrix(); glTranslatef(tx,8.5,5.5); sph(1.05,14); glPopMatrix()
        glColor3f(.88,.75,.1)
        box(tx-.06,9.5,5.44,tx+.06,11.0,5.56)
        box(tx-.45,10.4,5.44,tx+.45,10.6,5.56)
    glColor3f(.93,.90,.80); box(-4.6,0,5.95,4.6,5.6,6.2)
    glColor3f(.28,.16,.06); box(-1.0,0,6.18,1.0,2.8,6.22)
    glColor3f(1.0,.6,.1)
    glPushMatrix(); glTranslatef(0,4.2,6.22); glRotatef(90,0,1,0)
    q=gluNewQuadric(); gluCylinder(q,.65,.65,.05,20,1); glPopMatrix()
    glColor3f(.70,.68,.65)
    for s in range(4):
        e=s*.35; box(-4.5-e,s*.15,6.2,4.5+e,(s+1)*.15,6.2+e+.35)
    glPopMatrix()

def draw_escuela(x,z,rot=0):
    glPushMatrix(); glTranslatef(x,0,z); glRotatef(rot,0,1,0)
    glColor3f(.95,.90,.72); box(-8.0,0,-5.0,8.0,4.5,5.0)
    glColor3f(.28,.55,.28); box(-8.05,0,4.99,8.05,.70,5.08)
    glColor3f(.90,.85,.68); box(-8.2,4.5,-5.2,8.2,4.7,5.2)
    glColor3f(.85,.80,.62)
    box(-8.2,4.7,-5.2,8.2,5.2,-4.9); box(-8.2,4.7,4.9,8.2,5.2,5.2)
    box(-8.2,4.7,-5.2,-7.9,5.2,5.2); box(7.9,4.7,-5.2,8.2,5.2,5.2)
    glColor3f(.97,.95,.88)
    for cx in [-2.0,0.0,2.0]:
        glPushMatrix(); glTranslatef(cx,0,5.01); glRotatef(-90,1,0,0)
        cyl(.22,4.5,10); glPopMatrix()
    for wx in [-6.0,-3.5,3.5,6.0]:
        glColor3f(.55,.73,.88); box(wx-.55,1.0,5.00,wx+.55,2.5,5.10)
        glColor3f(.88,.82,.65); box(wx-.60,.95,5.06,wx+.60,2.55,5.09)
    glColor3f(.22,.45,.22); box(-.70,0,5.00,.70,2.5,5.09)
    glColor3f(.15,.30,.15); box(-3.5,3.5,5.05,3.5,4.2,5.12)
    glColor3f(.60,.60,.60); box(-7.8,0,4.0,-7.6,7.5,4.2)
    glColor3f(0,.55,.27); box(-7.6,6.8,4.1,-6.0,7.5,4.15)
    glColor3f(.95,.95,.95); box(-6.0,6.8,4.1,-4.4,7.5,4.15)
    glColor3f(.80,0,0); box(-4.4,6.8,4.1,-2.8,7.5,4.15)
    glPopMatrix()

def draw_fuente(x,z):
    glPushMatrix(); glTranslatef(x,0,z)
    glColor3f(.72,.70,.65)
    glPushMatrix(); glRotatef(-90,1,0,0); cyl(1.8,0.5,20); glPopMatrix()
    glColor3f(.30,.55,.80)
    glBegin(GL_POLYGON)
    for i in range(30):
        a=math.radians(i*12); glVertex3f(1.55*math.cos(a),.48,1.55*math.sin(a))
    glEnd()
    glColor3f(.78,.76,.70)
    glPushMatrix(); glTranslatef(0,.5,0); glRotatef(-90,1,0,0); cyl(.18,1.8,12); glPopMatrix()
    glPushMatrix(); glTranslatef(0,2.3,0); glRotatef(-90,1,0,0); cyl(.75,0.35,16); glPopMatrix()
    glPopMatrix()

def draw_jardin(x0,z0,x1,z1):
    glColor3f(.13,.48,.14)
    glBegin(GL_QUADS)
    glVertex3f(x0,.01,z0); glVertex3f(x1,.01,z0)
    glVertex3f(x1,.01,z1); glVertex3f(x0,.01,z1)
    glEnd()
    rng=random.Random(77); fcols=[(1,0,0),(1,.8,0),(1,0,.8),(1,1,0),(1,.4,0)]
    for _ in range(60):
        fx=rng.uniform(x0+.3,x1-.3); fz=rng.uniform(z0+.3,z1-.3)
        glColor3f(*fcols[rng.randint(0,4)])
        glPushMatrix(); glTranslatef(fx,.05,fz); sph(.09,6); glPopMatrix()
        glColor3f(.1,.5,.1)
        glPushMatrix(); glTranslatef(fx,0,fz); glRotatef(-90,1,0,0); cyl(.02,.10,5); glPopMatrix()

def draw_adoquin(x0,z0,x1,z1,w=2.5):
    dx,dz=x1-x0,z1-z0
    length=math.sqrt(dx*dx+dz*dz)
    ang=math.degrees(math.atan2(dx,dz))
    glPushMatrix(); glTranslatef(x0,.005,z0); glRotatef(-ang,0,1,0)
    glColor3f(.50,.48,.45); box(-w/2,0,0,w/2,.02,length)
    nr=int(length/.55); nc=int(w/.55)
    rng2=random.Random(99)
    for r in range(nr):
        for c in range(nc):
            g=.44+rng2.uniform(-.04,.04)
            glColor3f(g,g-.01,g-.02)
            rx=-w/2+c*.55+.04; rz=r*.55+.04
            box(rx,.005,rz,rx+.50,.025,rz+.50)
    glPopMatrix()

def draw_negocio(x,z,rot=0,color=(.88,.72,.35),lc=(.6,.05,.05)):
    glPushMatrix(); glTranslatef(x,0,z); glRotatef(rot,0,1,0)
    glColor3f(*color); box(-2.2,0,-1.5,2.2,3.2,1.5)
    glColor3f(.35,.08,0); box(-2.21,0,1.49,2.21,.5,1.58)
    glColor3f(*lc); box(-2.4,2.5,1.4,2.4,2.7,2.4)
    glColor3f(.65,.82,.92); box(-1.5,.5,1.49,1.5,2.3,1.56)
    glColor3f(.28,.16,.06); box(-.4,0,1.49,.4,2.2,1.57)
    glPopMatrix()

def draw_mountains():
    peaks=[(-80,35,-95),(0,45,-100),(80,30,-90),(-120,25,-85),(120,28,-88),(-50,20,-80),(50,22,-80)]
    for mx,mh,mz in peaks:
        glColor3f(.28,.38,.22)
        glBegin(GL_TRIANGLES)
        glVertex3f(mx-30,0,mz+15); glVertex3f(mx+30,0,mz+15); glVertex3f(mx,mh,mz)
        glEnd()
        glColor3f(.92,.93,.95)
        glBegin(GL_TRIANGLES)
        glVertex3f(mx-8,mh*.72,mz+4); glVertex3f(mx+8,mh*.72,mz+4); glVertex3f(mx,mh,mz)
        glEnd()

def draw_clouds():
    data=[(-20,22,-60),(10,25,-70),(35,20,-55),(-40,24,-75),(55,23,-65)]
    glColor3f(.97,.97,.97)
    for cx,cy,cz in data:
        for dx,dz,r in [(0,0,2.2),(-2,0,1.5),(2,0,1.5),(0,.8,1.3)]:
            glPushMatrix(); glTranslatef(cx+dx,cy,cz+dz); sph(r,10); glPopMatrix()

hud_tex_id = None
def init_hud_texture():
    global hud_tex_id
    hud_tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, hud_tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

def draw_hud_camera():
    with cam_frame_lock:
        if cam_frame_latest is None:
            return
        frame = cam_frame_latest.copy()
    h, w = frame.shape[:2]
    rgb = np.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0).copy()
    glBindTexture(GL_TEXTURE_2D, hud_tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, 1000, 0, 700, -1, 1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glEnable(GL_TEXTURE_2D); glDisable(GL_DEPTH_TEST)
    HW, HH = 240, 180
    glColor3f(1,1,1)
    glBegin(GL_QUADS)
    glTexCoord2f(0,0); glVertex2f(0,  0)
    glTexCoord2f(1,0); glVertex2f(HW, 0)
    glTexCoord2f(1,1); glVertex2f(HW, HH)
    glTexCoord2f(0,1); glVertex2f(0,  HH)
    glEnd()
    glDisable(GL_TEXTURE_2D); glEnable(GL_DEPTH_TEST)
    glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_pueblo():
    t = T()
    apply_camera_movement()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    yr = math.radians(cam_yaw)
    pr = math.radians(cam_pitch)
    lx = cam_x + math.cos(pr)*(-math.sin(yr))
    ly = cam_y + math.sin(pr)
    lz = cam_z + math.cos(pr)*(-math.cos(yr))
    gluLookAt(cam_x, cam_y, cam_z, lx, ly, lz, 0, 1, 0)
    draw_clouds()
    draw_mountains()
    glColor3f(.38,.36,.33)
    glBegin(GL_QUADS)
    glVertex3f(-100,0,100); glVertex3f(100,0,100)
    glVertex3f(100,0,-100); glVertex3f(-100,0,-100)
    glEnd()
    draw_jardin(-11,-11,11,11)
    draw_adoquin(0,11,0,50); draw_adoquin(0,-11,0,-50)
    draw_adoquin(11,0,50,0); draw_adoquin(-11,0,-50,0)
    draw_adoquin(11,11,45,45,w=2.2); draw_adoquin(-11,11,-45,45,w=2.2)
    draw_adoquin(11,-11,45,-45,w=2.2); draw_adoquin(-11,-11,-45,-45,w=2.2)
    draw_kiosk(0,0)
    draw_fuente(0,-7.5); draw_fuente(0,7.5)
    draw_iglesia(0,-42)
    draw_escuela(35,40,rot=180)
    for i,hx in enumerate(range(-28,32,14)):
        draw_house(hx,18,rot=180,w=3.5,d=2.5,sec=(hx%14==0))
        draw_house(hx,-18,rot=0,w=3.5,d=2.5,wc=(.92,.90,.82),rc=(.68,.18,.06))
    for hz in range(-26,20,13):
        draw_house(20,hz,rot=270,w=3.0,d=2.2,wc=(.93,.88,.78),rc=(.65,.15,.08))
        draw_house(-20,hz,rot=90,w=3.0,d=2.2,wc=(.88,.85,.75))
    for i,hx in enumerate(range(-45,50,15)):
        draw_house(hx,35,rot=180,w=4.0,d=2.8,sec=(i%2==0))
        draw_house(hx,-35,rot=0,w=4.0,d=2.8)
    for i,hz in enumerate(range(-32,36,14)):
        draw_house(40,hz,rot=270,w=3.5,d=2.4)
        draw_house(-40,hz,rot=90,w=3.5,d=2.4,wc=(.90,.87,.76))
    draw_negocio(15,16,rot=180,color=(.85,.70,.30),lc=(.1,.35,.1))
    draw_negocio(-15,16,rot=180,color=(.80,.65,.28),lc=(.6,.05,.05))
    draw_negocio(15,-16,rot=0,color=(.78,.60,.25),lc=(.1,.10,.55))
    draw_negocio(-15,-16,rot=0,color=(.82,.68,.28),lc=(.55,.35,.0))
    for i,(ax,az) in enumerate([(-8,-8),(8,-8),(-8,8),(8,8),(-3,-9),(3,-9),(-3,9),(3,9)]):
        draw_tree(ax,az,h=3.0+i%2,kind=i%2)
    for zz in range(-45,46,8):
        draw_tree(14,zz,h=4.0,kind=0); draw_tree(-14,zz,h=3.5+((zz+45)%3)*.5,kind=1)
    for xx in range(-45,46,8):
        draw_tree(xx,13,h=4.0,kind=1); draw_tree(xx,-13,h=3.5,kind=0)
    for ax,az,k in [(-30,25,0),(30,25,1),(-30,-25,1),(30,-25,0),
                     (-50,0,0),(50,0,1),(0,50,0),(0,-50,1),
                     (22,42,0),(-22,42,1),(22,-42,0),(-22,-42,1)]:
        draw_tree(ax,az,h=4.5,kind=k)
    pn=[(-12,14),(-6,14),(0,14),(6,14),(12,14)]
    ps=[(-12,-14),(-6,-14),(0,-14),(6,-14),(12,-14)]
    pe=[(13,-12),(13,-6),(13,0),(13,6),(13,12)]
    po=[(-13,-12),(-13,-6),(-13,0),(-13,6),(-13,12)]
    for px,pz in pn+ps+pe+po: draw_poste(px,pz)
    for i in range(len(pn)-1):
        draw_cable_banderines(pn[i][0],pn[i][1],pn[i+1][0],pn[i+1][1])
        draw_cable_banderines(ps[i][0],ps[i][1],ps[i+1][0],ps[i+1][1])
    for i in range(len(pe)-1):
        draw_cable_banderines(pe[i][0],pe[i][1],pe[i+1][0],pe[i+1][1],nb=12)
        draw_cable_banderines(po[i][0],po[i][1],po[i+1][0],po[i+1][1],nb=12)
    for zz in [-10,-5,0,5,10]:
        draw_cable_banderines(-13,zz,13,zz,nb=14)
    draw_car(5,30,rot=0,color=(.15,.25,.65))
    draw_car(-5,30,rot=180,color=(.7,.1,.1))
    draw_car(16,5,rot=90,color=(.1,.5,.15))
    draw_car(16,-8,rot=270,color=(.7,.6,.1))
    draw_car(0,-32,rot=0,color=(.5,.1,.5))
    draw_car(-18,10,rot=90,color=(.15,.15,.15))
    draw_car(3,50,rot=180,color=(.85,.40,.10))
    for sx,sz,d0,spd,radio,shirt in PERSONAS_DATA:
        px,pz,face=animated_pos(sx,sz,d0,spd,radio,t)
        draw_person_at(px,pz,face_deg=face,anim_t=t*spd*20,shirt=shirt)
    for sx,sz,d0,spd,radio in BURROS_DATA:
        px,pz,face=animated_pos(sx,sz,d0,spd,radio,t)
        draw_burro_at(px,pz,face_deg=face,anim_t=t*spd*20)
    for ad in [0,45,90,135,180,225,270,315]:
        a=math.radians(ad); draw_bench(9.5*math.cos(a),9.5*math.sin(a),rot=ad+90)
    draw_hud_camera()
    glfw.swap_buffers(window)

def main():
    global window
    vt = threading.Thread(target=vision_thread, daemon=True)
    vt.start()
    if not glfw.init(): return
    window = glfw.create_window(1000, 700, "Pueblo Magico Michoacan", None, None)
    if not window: glfw.terminate(); return
    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)
    init_gl()
    init_hud_texture()


    while not glfw.window_should_close(window):
        draw_pueblo()
        glfw.poll_events()
    glfw.terminate()

if __name__ == "__main__":
    main()
