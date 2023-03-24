import cv2
import numpy as np

kemalSunal = cv2.imread("indir.jpg")
kemalSunal[:,:,1] = 255  #0 blue kısmı ifade ediyor ve oraya 255 verince blue efekt veriyor
#1 olunca green 2 oluncada red efekt veriyor.
kemalSunal[:,:,2] = 150
cv2.imshow("Kemal Sunal", kemalSunal)



cv2.waitKey(0)
cv2.destroyAllWindows()