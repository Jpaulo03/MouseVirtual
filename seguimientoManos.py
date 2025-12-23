# seguimientoManos.py
import math
import cv2
import mediapipe as mp
import time


class detectormanos():
    def __init__(self, mode=False, maxManos=2, Confedteccion=0.5, Confsegui=0.5):
        self.mode = mode
        self.maxManos = maxManos
        self.Confedteccion = Confedteccion
        self.Confsegui = Confsegui

        self.mpmanos = mp.solutions.hands
        self.manos = self.mpmanos.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxManos,
            min_detection_confidence=self.Confedteccion,
            min_tracking_confidence=self.Confsegui
        )

        self.dibujo = mp.solutions.drawing_utils
        self.tip = [4, 8, 12, 16, 20]
        self.resultados = None
        self.lista = []

    # Encontrar manos 
    def encontrarmanos(self, frame, dibujar=True):
        imgcolor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.resultados = self.manos.process(imgcolor)

        if self.resultados and self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.dibujo.draw_landmarks(
                        frame,
                        mano,
                        self.mpmanos.HAND_CONNECTIONS
                    )

        return frame

    def encontrarposicion(self, frame, ManoNum=0, dibujar=True):
        xlista = []
        ylista = []
        bbox = []
        self.lista = []

        if self.resultados and self.resultados.multi_hand_landmarks:
            miMano = self.resultados.multi_hand_landmarks[ManoNum]
            for id, lm in enumerate(miMano.landmark):
                alto, ancho, c = frame.shape
                cx, cy = int(lm.x * ancho), int(lm.y * alto)
                xlista.append(cx)
                ylista.append(cy)
                self.lista.append([id, cx, cy])

                if dibujar:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)

            xmin, xmax = min(xlista), max(xlista)
            ymin, ymax = min(ylista), max(ylista)
            bbox = xmin, ymin, xmax, ymax

            if dibujar:
                cv2.rectangle(frame, (xmin - 20, ymin - 20),
                              (xmax + 20, ymax + 20), (0, 255, 0), 2)
        return self.lista, bbox

    def dedosarriba(self):
        dedos = []

        # pulgar (tip 4)
        if self.lista[self.tip[0]][1] > self.lista[self.tip[0] - 1][1]:
            dedos.append(1)
        else:
            dedos.append(0)

        # 4 dedos restantes
        for id in range(1, 5):
            if self.lista[self.tip[id]][2] < self.lista[self.tip[id] - 2][2]:
                dedos.append(1)
            else:
                dedos.append(0)

        return dedos

    def distancia(self, p1, p2, frame, dibujar=True, r=15, t=3):
        x1, y1 = self.lista[p1][1:]
        x2, y2 = self.lista[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if dibujar:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, frame, [x1, y1, x2, y2, cx, cy]

    def obtener_labels(self):
        """
        Retorna una lista con las etiquetas ('Left' o 'Right') en el mismo orden
        en que aparecen self.resultados.multi_hand_landmarks.
        """
        labels = []
        if self.resultados and self.resultados.multi_handedness:
            for mano_h in self.resultados.multi_handedness:
                labels.append(mano_h.classification[0].label)
        return labels


def main():
    ptiempo = 0
    ctiempo = 0

    cap = cv2.VideoCapture(0)
    detector = detectormanos()

    while True:
        ret, frame = cap.read()
        frame = detector.encontrarmanos(frame)
        lista, bbox = detector.encontrarposicion(frame)

        if len(lista) != 0:
            print(lista[4]) 

        ctiempo = time.time()
        fps = 1 / (ctiempo - ptiempo) if (ctiempo - ptiempo) != 0 else 0
        ptiempo = ctiempo

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 70),
                      cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Manos", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
