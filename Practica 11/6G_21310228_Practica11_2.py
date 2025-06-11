import cv2

cap = cv2.VideoCapture(0)  # o usa 'video.mp4'

# Leer primer frame como fondo inicial
ret, fondo = cap.read()
fondo_gray = cv2.cvtColor(fondo, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Diferencia absoluta entre fondo y frame actual
    diff = cv2.absdiff(fondo_gray, gray)

    # Umbral para mostrar sólo movimiento
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    cv2.imshow('Video Original', frame)
    cv2.imshow('Movimiento detectado', thresh)

    if cv2.waitKey(1) & 0xFF == 27:  # Salir con ESC
        break

cap.release()
cv2.destroyAllWindows()