import cv2
import numpy as np

kamera = cv2.VideoCapture("video.mp4") #0 kendi pc mizdeki 1 yazarsak usb ile takılan kamera 2 yazarsak bir video yüklersek dosyaya o videodan görüntüleri alır
while True:
    ret, goruntu  = kamera.read()           
    cv2.imshow("volkan", goruntu)

    if cv2.waitKey(30) & 0xFF == ("q"):
        break

kamera.release()

cv2.destroyAllWindows()
