# MouseVirtual.py
import cv2
import numpy as np
import seguimientoManos as sm
import pyautogui
import math
import time

# --------------------------
# CONFIG: cambia aquí según tu cámara externa
# Si es una webcam USB normalmente es 1 o 2 (prueba 0,1,2)
# También puedes usar una URL si tu cámara lo permite (e.g. "http://ip:port/stream")
CAM_INDEX = 1
# --------------------------

anchocam, altocam = 640, 480
cuadro = 100
anchopanta, altopanta = pyautogui.size()
sua = 5  # suavizado de movimiento

# variables para suavizado
pubix, pubiy = 0, 0
cubix, cubiy = 0, 0

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(3, anchocam)
cap.set(4, altocam)

detector = sm.detectormanos(maxManos=2)

# variables de volumen visual
volBar = 400

# pequeño delay para estabilizar la cam
time.sleep(1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer frame. Revisa CAM_INDEX o la conexión de la cámara.")
        break

    frame = detector.encontrarmanos(frame, dibujar=True)

    # obtenemos las etiquetas 'Left'/'Right' en orden detectado
    labels = detector.obtener_labels()

    if detector.resultados and detector.resultados.multi_hand_landmarks:
        # recorremos cada mano detectada
        for i, label in enumerate(labels):
            lista, bbox = detector.encontrarposicion(frame, ManoNum=i, dibujar=False)

            if len(lista) == 0:
                continue

            # detectamos dedos levantados para esta mano
            dedos = detector.dedosarriba()

            # --- Mano derecha: mover mouse + clicks ---
            # Por convención, Right => puntero & clicks
            if label == 'Right':
                # posición del índice (landmark 8)
                x1, y1 = lista[8][1:]

                # rectángulo de referencia
                cv2.rectangle(frame, (cuadro, cuadro),
                              (anchocam - cuadro, altocam - cuadro),
                              (0, 255, 0), 2)

                # MOVER MOUSE: solo dedo índice levantado
                if dedos[1] == 1 and sum(dedos) == 1:
                    x3 = np.interp(x1, (cuadro, anchocam - cuadro), (0, anchopanta))
                    y3 = np.interp(y1, (cuadro, altocam - cuadro), (0, altopanta))

                    cubix = pubix + (x3 - pubix) / sua
                    cubiy = pubiy + (y3 - pubiy) / sua

                    # invertir eje x si quieres que coincida con tu pantalla
                    pyautogui.moveTo(anchopanta - cubix, cubiy)

                    cv2.circle(frame, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                    pubix, pubiy = cubix, cubiy

                # CLICK IZQUIERDO: medio (12) + anular (16) juntos
                longitud_izq, frame, info_izq = detector.distancia(12, 16, frame, dibujar=False)
                if dedos[2] == 1 and dedos[3] == 1 and sum(dedos) == 2 and longitud_izq < 35:
                    cv2.circle(frame, (info_izq[4], info_izq[5]), 12, (0, 255, 0), cv2.FILLED)
                    pyautogui.click()

                # CLICK DERECHO: anular (16) + meñique (20) juntos
                longitud_der, frame, info_der = detector.distancia(16, 20, frame, dibujar=False)
                if dedos[3] == 1 and dedos[4] == 1 and sum(dedos) == 2 and longitud_der < 35:
                    cv2.circle(frame, (info_der[4], info_der[5]), 12, (255, 0, 255), cv2.FILLED)
                    pyautogui.rightClick()

            # --- Mano izquierda: volumen y scroll ---
            # Por convención, Left => volumen y scroll
            elif label == 'Left':
                # SCROLL ARRIBA: los 5 dedos levantados
                if sum(dedos) == 5:
                    pyautogui.scroll(5)
                    cv2.putText(frame, "SCROLL ARRIBA", (anchocam - 200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                # SCROLL ABAJO: cerrar el puño (ningún dedo)
                if sum(dedos) == 0:
                    pyautogui.scroll(-5)
                    cv2.putText(frame, "SCROLL ABAJO", (anchocam - 200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                # CONTROL DE VOLUMEN: Pulgar (4) + índice (8)
                if dedos[0] == 1 and dedos[1] == 1 and sum(dedos) == 2:
                    longitud_vol, frame, info_vol = detector.distancia(4, 8, frame, dibujar=False)

                    # umbrales simples para subir/bajar
                    if longitud_vol > 150:
                        pyautogui.press('volumeup')
                        cv2.putText(frame, "VOLUMEN +", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                    elif longitud_vol < 50:
                        pyautogui.press('volumedown')
                        cv2.putText(frame, "VOLUMEN -", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

                    # barra visual
                    volBar = np.interp(longitud_vol, [30, 250], [400, 150])
                    cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
                    cv2.rectangle(frame, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

    # FPS
    cv2.imshow("Mouse Virtual Algoritmica", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
