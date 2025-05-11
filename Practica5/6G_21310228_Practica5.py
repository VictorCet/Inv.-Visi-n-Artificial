import cv2
import numpy as np

# Cargar imagen
img = cv2.imread('descarga.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Umbrales fijos
_, th_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, th_binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
_, th_trunc = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
_, th_tozero = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
_, th_tozero_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)

# Umbral adaptativo
th_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 11, 2)
th_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

# Otsu
_, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Mostrar resultados
cv2.imshow('Original', img)
cv2.imshow('Gray', gray)
cv2.imshow('Binary', th_binary)
cv2.imshow('Binary Inv', th_binary_inv)
cv2.imshow('Trunc', th_trunc)
cv2.imshow('To Zero', th_tozero)
cv2.imshow('To Zero Inv', th_tozero_inv)
cv2.imshow('Adapt Mean', th_mean)
cv2.imshow('Adapt Gauss', th_gauss)
cv2.imshow('Otsu', th_otsu)

cv2.waitKey(0)
cv2.destroyAllWindows()
