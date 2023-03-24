import cv2
import numpy as np 

resim = cv2.imread("logo.png")  #resmi okur , 0 değeri renkli resmi renksiz hale getirdi.


cv2.imshow("NVDIA", resim)
# print(resim) #resmin 0,255 değerleri arasındaki matrislerini verdi
print(resim.size) #resmin boyutu
print(resim.shape)  #Genişlik, yükseklik ve kaç kanal oldugu

cv2.waitKey(0)
cv2.destroyAllWindows()
