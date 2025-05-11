import cv2
import numpy as np

# Iniciar captura de video (cámara)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir imagen a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Filtro rojo (hay que hacer dos rangos por el "bucle" del rojo en HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # Filtro verde
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Filtro azul
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Aplicar máscaras
    red_result = cv2.bitwise_and(frame, frame, mask=mask_red)
    green_result = cv2.bitwise_and(frame, frame, mask=mask_green)
    blue_result = cv2.bitwise_and(frame, frame, mask=mask_blue)

    # Mostrar resultados
    cv2.imshow('Original', frame)
    cv2.imshow('Rojo', red_result)
    cv2.imshow('Verde', green_result)
    cv2.imshow('Azul', blue_result)

    if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC para salir #
        break

cap.release()
cv2.destroyAllWindows()
