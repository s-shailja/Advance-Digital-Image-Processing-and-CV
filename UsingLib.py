import cv2
import numpy as np
import matplotlib.pyplot as plt

img1=cv2.imread('cap.bmp',-1)
img2=cv2.imread('lego.tif',-1)
#Conversion of color imge to gray image using library
im = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
cv2.imshow('RGB to Gray',im)
img=cv2.imread('sp_noise.jpg',0) 
#Mean Filter 
blur = cv2.blur(img,(3,3))
cv2.imshow('Mean Filter Denoising',blur)
#Median Filter
median = cv2.medianBlur(img,5)
cv2.imshow('Median Filtering Denoising',median)
cv2.waitKey(0)
cv2.destroyWindow('Mean Filter Denoising')
cv2.destroyWindow('Median Filtering Denoising')
