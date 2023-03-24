import cv2 
import numpy as np

wolfimage = cv2.imread("kurt.jpg")
cv2.imshow("Wolf",wolfimage)
print(wolfimage[(200,80)]) #Resmin sol köşesi baz alınarak aşağı 200 30 pikselde sağ bgr değeri için
print("Resmin boyutu: "+ str(wolfimage.size))
print("Resmin Özellikleri: "+ str(wolfimage.shape))
print("Resmin Veri Tipi: "+ str(wolfimage.dtype))

cv2.waitKey(0)
cv2.destroyAllWindows()


