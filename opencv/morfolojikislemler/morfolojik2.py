import cv2
import numpy as np

image = cv2.imread("sch.png")
kernel = np.ones((5,5),np.uint8)

erosion = cv2.erode(image,kernel,iterations=1)
dilation = cv2.dilate(erosion,kernel,iterations=1)

#dilation = cv2.dilate(image,kernel,iterations=1) #dilation işlemi beyaz bölgeyi arttırdı.
#erosion = cv2.erode(image,kernel,iterations=1) #dilationun tam tersi, kirliliği azaltır.

cv2.imshow("orijinal",image)
cv2.imshow("Dilation",dilation)
cv2.imshow("Erosion",erosion)

cv2.waitKey(0)
cv2.destroyAllWindows()