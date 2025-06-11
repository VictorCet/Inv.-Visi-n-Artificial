import cv2

img_template = cv2.imread('template.png', 0)  # Plantilla
img_scene = cv2.imread('descarga.png', 0)    # Principal

orb = cv2.ORB_create() # Detector ORB

kp_template, des_template = orb.detectAndCompute(img_template, None) # Detectar keypoints descriptores
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)# Compara descriptores Brute_Force matcher
matches = bf.match(des_template, des_scene)

matches = sorted(matches, key=lambda x: x.distance) # Ordena coincidencias

resultado = cv2.drawMatches(img_template, kp_template, img_scene, kp_scene, matches[:20], None, flags=2)# Dibuja mejores coincidencias

cv2.imshow('Similitudes detectadas', resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()
