import cv2

gok = cv2.imread("gok.jpg",0) #0 gri tonlamalı getiriyor fotoğrafı

ret,thresh1 = cv2.threshold(gok,127,255,cv2.THRESH_BINARY) #simplethresholding
ret,thresh2 = cv2.threshold(gok,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(gok,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(gok,127,255,cv2.THRESH_TOZERO)



cv2.imshow("original",gok)
cv2.imshow("thresh1",thresh1)
cv2.imshow("thresh2",thresh2)
cv2.imshow("thresh3",thresh3)
cv2.imshow("thresh4",thresh4)

cv2.waitKey(0)
cv2.destroyAllWindows()


