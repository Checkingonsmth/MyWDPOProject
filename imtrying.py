import json
from pathlib import Path
from typing import Dict

import numpy as np
import click
import cv2
from tqdm import tqdm

img = cv2.imread('data/15.jpg', cv2.IMREAD_COLOR)
cv2.namedWindow('image')


# g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gaussian_img = cv2.GaussianBlur(g_img, (9, 9), 0)
# canny_img = cv2.Canny(gaussian_img, 50, 80)
# dilated_img = cv2.dilate(canny_img, (1, 1), iterations=10)
# (cnt, hier) = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gaussian_img = cv2.GaussianBlur(hsv, (11, 11), 3)
#threshold dla zielonego
threshold = cv2.inRange(hsv, (33,15,50), (70, 250,240))
dilated_img = cv2.erode(threshold, (20, 20), iterations=2)
canny_img = cv2.Canny(dilated_img, 0, 255,100)
(cnt, hier) = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, cnt, -1, (0,255,0), 3)
resized= cv2.resize(hsv, (1280,700))
x=0
for i in range(len(cnt)):
    if (cv2.contourArea(cnt[i]) >200):
        x=x+1

print(x)
while True:
    cv2.imshow('image',resized)
    cv2.waitKey(100)

#contour.area
#zolty 24-26, 10-30
