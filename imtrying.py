import json
from pathlib import Path
from typing import Dict

import numpy as np
import click
import cv2
from tqdm import tqdm

img = cv2.imread('data/03.jpg', cv2.IMREAD_COLOR)
cv2.namedWindow('image')


# g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gaussian_img = cv2.GaussianBlur(g_img, (9, 9), 0)
# canny_img = cv2.Canny(gaussian_img, 50, 80)
# dilated_img = cv2.dilate(canny_img, (1, 1), iterations=10)
# (cnt, hier) = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gaussian_img = cv2.GaussianBlur(hsv, (11, 11), 3)
#threshold dla zielonego
# threshold = cv2.inRange(hsv, (36,110,110), (51, 250,240))
# threshold = cv2.inRange(hsv, (36,110,110), (50, 250,240))
#tu bylo erode i iterations=3
#cntarea=200

#threshold dla fioletowego
# threshold = cv2.inRange(hsv, (160,65,50), (170, 200,200))
# threshold = cv2.inRange(hsv, (135,69,50), (170, 170,100))
#cnt area=400
#dilate i iterations=2

#threshold dla zołtego
# threshold = cv2.inRange(hsv, (22,180,180), (30, 255,250))
#cnt area=200
#erode i iterations 3

#threshold dla czerwonego
threshold1 = cv2.inRange(hsv, (1,65,85), (8, 255,255))
threshold2 = cv2.inRange(hsv, (178,65,85), (180, 255,255))
threshold=threshold2+threshold1
dilated_img = cv2.dilate(threshold, (50, 50), iterations=2)
canny_img = cv2.Canny(dilated_img, 0, 255,100)
(cnt, hier) = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, cnt, -1, (0,255,0), 3)

x=0

size0=4080
size1=3072
scale=1

size0v=500
size1v=1000
if (img.shape[0]!=size0 & img.shape[1]!=size0):
    if (img.shape[0]>img.shape[1]):
        scale=img.shape[0]/size0
    else:
        scale = img.shape[1]/size0


if (img.shape[0]>img.shape[1]):
    resized = cv2.resize(img, (size0v, size1v))
else:
    resized = cv2.resize(img, (size1v, size0v))


for i in range(len(cnt)):
    if (cv2.contourArea(cnt[i]) >(700*scale)):
        x=x+1


print(x)
print(img.shape[0])
print(img.shape[1])
while True:
    cv2.imshow('image',resized)
    cv2.waitKey(100)

#contour.area
#zolty 24-26, 10-30
#czerwony jest na koncu i poczatku- lepiej nalozyc dwa filtry- np. od 0 do 5 i od 170 do 180