import cv2
import numpy as np

bilge = cv2.imread("bilge.JPG")
resimgri = cv2.cvtColor(bilge,cv2.COLOR_BGRA2BGR)
yukseklik,genislik,kanalsayisi = bilge.shape
print("Orijinal",yukseklik,genislik,kanalsayisi)




cv2.imshow("orijinal",bilge)
cv2.imshow("grilestirilmis resim", resimgri)

cv2.waitKey(0)
cv2.destroyAllWindows()