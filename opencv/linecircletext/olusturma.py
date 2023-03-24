import cv2
import numpy as np

resim = np.zeros((500,500,3),dtype="uint8")
cv2.line(resim,(0,0),(150,150),(200,100,100),4)
cv2.circle(resim,(150,150),50,(200,150,200),2)
cv2.putText(resim,"Volkan ozkan",(0,300),cv2.FONT_HERSHEY_PLAIN,4,(200,25,255),3)



cv2.imshow("circle",resim)

cv2.waitKey(0)
cv2.destroyAllWindows()