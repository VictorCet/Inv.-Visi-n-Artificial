import cv2  #procesamiento de imágenes
import numpy as np  # operaciones numéricas

img_color = cv2.imread('descarga.png')  # Imagen original en color
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)  # Convierte a escala de grises para el análisis


x, y, w, h = 100, 100, 100, 100 #Region de interes
# Coordenadas del ROI: punto (x, y), ancho (w) y alto (h)
roi_color = img_color[y:y+h, x:x+w]  # Recortamos el ROI en color
roi_gray = img_gray[y:y+h, x:x+w]    # Recortamos el ROI en escala de grises

roi_gray = np.float32(roi_gray) #Deteccion de esquinas
dst = cv2.cornerHarris(roi_gray, blockSize=2, ksize=3, k=0.04)#Formato float32
# blockSize: tamaño del bloque considerado para detección
# ksize: tamaño del kernel de Sobel
# k: parámetro libre entre 0.04 y 0.06

dst = cv2.dilate(dst, None)#dilate para ver las esquinas mas claras

roi_color[dst > 0.01 * dst.max()] = [0, 0, 255] #Resalta las esquina en rojo

cv2.imshow('ROI con esquinas detectadas', roi_color) #Muestra el ROI con las esquinas detectadas
cv2.waitKey(0) 
cv2.destroyAllWindows() 