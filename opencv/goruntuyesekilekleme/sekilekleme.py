import cv2
import numpy as np



kamera = cv2.VideoCapture("kacirma.mp4")
while True:
    ret, goruntu = kamera.read()
    cv2.circle(goruntu,(900,300),100,(200,150,200),2)
    cv2.putText(goruntu,"Volkan",(0,150),cv2.FONT_HERSHEY_PLAIN,2,(200,25,255),3)

    
    cv2.imshow("Volkan",goruntu)


    if cv2.waitKey(25) & 0xFF == ("q"):
        break
kamera.release()

cv2.destroyAllWindows()