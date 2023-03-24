import cv2
import numpy as np 

resim = cv2.imread("logo.png",0)  #resmi okur , 0 değeri renkli resmi renksiz hale getirdi.


cv2.imshow("BJK", resim)
cv2.imwrite("renksizlogo.png",resim) #renksiz logoyu dosyaya kaydettik
cv2.waitKey(0)
cv2.destroyAllWindows()

