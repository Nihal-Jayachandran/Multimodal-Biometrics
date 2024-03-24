import cv2
image = cv2.imread('finger1.bmp',0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
topHat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)# Black Hat Transform
blackHat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
res = image + topHat - blackHat
cv2.imshow("frame",res)
cv2.waitKey(0)