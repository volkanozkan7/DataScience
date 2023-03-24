import cv2
import numpy as np

ronaldo = cv2.imread("ronaldo.png")
aynalama = cv2.copyMakeBorder(ronaldo,300,300,400,400,cv2.BORDER_REFLECT)
sarilan = cv2.copyMakeBorder(ronaldo,300,300,400,400,cv2.BORDER_CONSTANT
                             value= (0,0,255))


cv2.imshow("Ronaldo", aynalama)



cv2.waitKey(0)
cv2.destroyAllWindows()