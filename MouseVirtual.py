import cv2
import numpy as np
import seguimientoManos as sm
import pyautogui 
import math

# ------------------------- Declaración de variables -------------------------
anchocam, altocam = 640, 480
cuadro = 100
anchopanta, altopanta = pyautogui.size() 
sua = 5

pubix, pubiy = 0, 0
cubix, cubiy = 0, 0

# Variables para visualización de Volumen 
vol = 0
volBar = 400
volPer = 0

# ------------------------- Configuración de cámara -------------------------
cap = cv2.VideoCapture(0)
cap.set(3, anchocam)
cap.set(4, altocam)

# ------------------------- Detector -------------------------
detector = sm.detectormanos(maxManos=1)

while True:
    ret, frame = cap.read()
    frame = detector.encontrarmanos(frame)
    lista, bbox = detector.encontrarposicion(frame, dibujar=False)

    if len(lista) != 0:
        x1, y1 = lista[8][1:]   # Índice (8)
        
        # Nuevos puntos para Click Izquierdo y Derecho:
        x_m, y_m = lista[12][1:]  # Medio (12)
        x_a, y_a = lista[16][1:]  # Anular (16)
        x_n, y_n = lista[20][1:]  # Meñique (20)
        
        dedos = detector.dedosarriba()

        # Zona de control
        cv2.rectangle(frame, (cuadro, cuadro),
                      (anchocam - cuadro, altocam - cuadro),
                      (0, 255, 0), 2)

        # ---------------------- 1. MOVER MOUSE: solo dedo índice levantado ----------------------
        # Mantenemos el movimiento con solo el Índice
        if dedos[1] == 1 and sum(dedos) == 1: 
            x3 = np.interp(x1, (cuadro, anchocam - cuadro), (0, anchopanta))
            y3 = np.interp(y1, (cuadro, altocam - cuadro), (0, altopanta))

            cubix = pubix + (x3 - pubix) / sua
            cubiy = pubiy + (y3 - pubiy) / sua

            pyautogui.moveTo(anchopanta - cubix, cubiy) 
            
            cv2.circle(frame, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            pubix, pubiy = cubix, cubiy

        # ---------------------- 1. CLICK IZQUIERDO: Medio + Anular (cerca) ----------------------
        # Usamos la distancia entre el Medio (12) y el Anular (16)
        longitud_izq, frame, info_izq = detector.distancia(12, 16, frame, dibujar=False)

        # Condición: Solo Medio (2) y Anular (3) levantados (para evitar confusión con 5 dedos o volumen)
        if dedos[2] == 1 and dedos[3] == 1 and sum(dedos) == 2 and longitud_izq < 35:
            cv2.circle(frame, (info_izq[4], info_izq[5]), 12, (0, 255, 0), cv2.FILLED)
            pyautogui.click() 

        # ---------------------- 1. CLICK DERECHO: Anular + Meñique (cerca) ----------------------
        # Usamos la distancia entre el Anular (16) y el Meñique (20)
        longitud_der, frame, info_der = detector.distancia(16, 20, frame, dibujar=False)

        # Condición: Solo Anular (3) y Meñique (4) levantados
        if dedos[3] == 1 and dedos[4] == 1 and sum(dedos) == 2 and longitud_der < 35:
            cv2.circle(frame, (info_der[4], info_der[5]), 12, (255, 0, 255), cv2.FILLED)
            pyautogui.rightClick() 
            
        # ---------------------- 2. SCROLL: 5 dedos hacia arriba (Se mantiene) ----------------------
        if sum(dedos) == 5:
            pyautogui.scroll(5) 
            cv2.putText(frame, "SCROLL ARRIBA", (anchocam - 200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        # ---------------------- 3. CONTROL DE VOLUMEN: Índice + Pulgar (Se mantiene) ----------------------
        if dedos[0] == 1 and dedos[1] == 1 and sum(dedos) == 2:
            longitud_vol, frame, info_vol = detector.distancia(4, 8, frame, dibujar=False)
            
            # Subir volumen (abriendo los dedos)
            if longitud_vol > 150: 
                pyautogui.press('volumeup') 
                cv2.putText(frame, "VOLUMEN +", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

            # Bajar volumen (uniendo los dedos)
            elif longitud_vol < 50:
                pyautogui.press('volumedown')
                cv2.putText(frame, "VOLUMEN -", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            
            # Dibujar barra de volumen
            volBar = np.interp(longitud_vol, [30, 250], [400, 150])
            cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(frame, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)


    cv2.imshow("Mouse Virtual", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()