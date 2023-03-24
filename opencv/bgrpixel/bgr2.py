import cv2 
import numpy as np

wolfimage = cv2.imread("kurt.jpg")
wolfimage[100,100] = [255,255,255]
for i in range(100):
     wolfimage[50,i] = [255,255,255]  #Belli noktayı beyaza boyadı

cv2.imshow("Wolf",wolfimage)


cv2.waitKey(0)
cv2.destroyAllWindows()