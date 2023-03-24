import cv2
import numpy as np

image = cv2.imread("bit.jpg")
siyah = cv2.imread("siyah.jpg")

bit_and = cv2.bitwise_and(image,siyah)
# aynı boyutlarda iki resim almak gerekiyor . o yüzden hata veriyor. beyazları 1 siyahı 0 alır.
cv2.imshow("image",image)
cv2.imshow("bitwise", bit_and)

cv2.waitKey(0)
cv2.destroyAllWindows()