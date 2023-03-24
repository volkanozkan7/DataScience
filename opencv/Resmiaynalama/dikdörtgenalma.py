import cv2
import numpy as np

bilge = cv2.imread("bilge.JPG")
##bilge[100:150,230:310,0] = 255 #100 150 y için 230 310 x için

cv2.rectangle(bilge,(300,700),(600,300),[0,0,255],3) #sondaki 3 çizgilerin kalınlıgı
cv2.imshow("Bilge", bilge)



cv2.waitKey(0)
cv2.destroyAllWindows()