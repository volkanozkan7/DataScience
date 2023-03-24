import cv2
import numpy as np

bilge = cv2.imread("bilge.JPG")
ikikat = cv2.pyrUp(bilge)
ikikatb = cv2.pyrDown(bilge)

cv2.imshow("orijinal",bilge)
cv2.imshow("ikikat", ikikat)
cv2.imshow("ikikatb",bilge)

cv2.waitKey(0)
cv2.destroyAllWindows()