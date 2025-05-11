import cv2
import numpy as np

# Cargar imagen
img = cv2.imread('descarga.png')

# ---------- DIBUJAR SOBRE LA IMAGEN ----------

# Dibujar un rectángulo
cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)

# Dibujar un círculo
cv2.circle(img, (300, 150), 40, (255, 0, 0), 2)

# Dibujar una línea
cv2.line(img, (100, 300), (400, 300), (0, 0, 255), 2)

# Escribir texto
cv2.putText(img, 'Region de Interes', (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# ---------- ROI (Región de Interés) ----------

# Definir coordenadas de la ROI (por ejemplo, el mismo rectángulo que dibujaste)
roi = img[50:200, 50:200]

# Mostrar la ROI en una ventana aparte
cv2.imshow('ROI', roi)

# Mostrar la imagen con los dibujos
cv2.imshow('Imagen con Dibujos', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
