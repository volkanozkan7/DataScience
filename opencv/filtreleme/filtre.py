import cv2

image = cv2.imread("fotograf.jpg")
meanfilter = cv2.blur(image,(3,3),)  #Mean filteringde bir ortalama alınır mesela burda 3 3 lük
meanfilter2 = cv2.blur(image,(6,6),)
cv2.imshow("Fotograf",image)
cv2.imshow("Mean Filter",meanfilter)
cv2.imshow("Mean Filter2",meanfilter2)
#Median filteringde ortada kalan değer alınır. 1,2,3,4,5 değerleri olsa 3 orta değer

medianfilter = cv2.medianBlur(image,3) #Meanfiltera göre daha net bir görüntü çıktı
cv2.imshow("Median",image)
gauss = cv2.GaussianBlur(image,(3,3),0)
cv2.imshow("Gauss",image)

cv2.waitKey(0)
cv2.destroyAllWindows()