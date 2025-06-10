import cv2  
import numpy as np  
import matplotlib.pyplot as plt  

img = cv2.imread('descarga.png', cv2.IMREAD_GRAYSCALE)

img_suavizada = cv2.GaussianBlur(img, (5, 5), 0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

tophat = cv2.morphologyEx(img_suavizada, cv2.MORPH_TOPHAT, kernel)

blackhat = cv2.morphologyEx(img_suavizada, cv2.MORPH_BLACKHAT, kernel)

plt.figure(figsize=(10, 6)) 

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original: samus.png')
plt.axis('off')  

plt.subplot(2, 2, 2)
plt.imshow(img_suavizada, cmap='gray')
plt.title('Suavizado (Filtro Gaussiano)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(tophat, cmap='gray')
plt.title('TopHat')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(blackhat, cmap='gray')
plt.title('BlackHat')
plt.axis('off')

plt.tight_layout()
plt.show() 