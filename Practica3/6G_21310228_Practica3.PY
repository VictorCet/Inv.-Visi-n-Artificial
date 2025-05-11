import cv2 #procesamiento de imagenes

import numpy as np # para calculos
# graficas e imagenes
import matplotlib.pyplot as plt

# Cargar la imagen original en escala de grises
img = cv2.imread('descarga.png', cv2.IMREAD_GRAYSCALE)

# Imagen modificada desde la práctica 2 (por ejemplo, negación)
img_modificada = cv2.bitwise_not(img) # esta es la imagen de práctica 2

# -------- Histograma original de la imagen modificada --------
hist_original = cv2.calcHist([img_modificada], [0], None, [256], [0, 256])

# -------- Ecualización de histograma --------
img_eq = cv2.equalizeHist(img_modificada)

# -------- Histograma ecualizado --------
hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

# -------- Mostrar todo en una sola ventana con matplotlib --------
plt.figure(figsize=(10, 6))

# Imagen modificada
plt.subplot(2, 2, 1)
plt.imshow(img_modificada, cmap='gray')
plt.title('Imagen de Práctica 2 (Negada)')
plt.axis('off') #version negada

# Histograma original
plt.subplot(2, 2, 2)
plt.plot(hist_original, color='black')
plt.title('Histograma Original')
plt.xlim([0, 256]) #como se distribuyen los valores de gris

# Imagen ecualizada
plt.subplot(2, 2, 3)
plt.imshow(img_eq, cmap='gray')
plt.title('Imagen Ecualizada')
plt.axis('off') 

# Histograma ecualizado
plt.subplot(2, 2, 4)
plt.plot(hist_eq, color='black')
plt.title('Histograma Ecualizado')
plt.xlim([0, 256]) #cambio en la distribucion de los pixeles

plt.tight_layout()
plt.show() #para ajustar los espacios entre graficas
