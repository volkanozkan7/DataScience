import cv2

kamera = cv2.VideoCapture("kacirma.mp4")
width = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(width,height)
fourcc = cv2.VideoWriter_fourcc(*"MP4V") #MP4 formatında kaydedicek
writer = cv2.VideoWriter("kayitdosyasi.mp4", fourcc,20,(width,height))


while True:
    ret, frame = kamera.read()
    writer.write(frame)
    cv2.imshow("kayit videosu",frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

kamera.release()
writer.release()
cv2.destroyAllWindows()