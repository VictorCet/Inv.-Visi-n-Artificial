import numpy as np
import cv2
imagen=cv2.imread('watch.jpg')
m,n,c=imagen.shape
imagenb=np.zeros((m,n))

for x in range(m):
    for y in range(n):
        if (43 < imagen[x,y,0] < 159) and (25 < imagen[x,y,1] < 150 ) and (13 < imagen[x,y,2] < 255):
            imagenb[x,y]=255

cv2.imshow('imagenb',imagenb)
cv2.imshow("watch",imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
