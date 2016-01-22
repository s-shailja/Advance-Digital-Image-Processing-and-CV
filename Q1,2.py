import numpy as np
import cv2
from cv2 import *
img1=cv2.imread('cap.bmp',-1)
img2=cv2.imread('lego.tif',-1)
def displayImages():	
	cv2.imshow("Image 1: ", img1)	
	cv2.imshow("Image 2:", img2)
	cv2.waitKey(0)
	cv2.destroyWindow("Image 1: ")
	cv2.destroyWindow("Image 2:")	
def RGBtoGray(image):	
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			[r,g,b]=image[i][j]
			image[i][j]=0.2126 * r + 0.7152 * g + 0.0722 * b
	return image

image=cv2.imread('lego.tif',-1)
image=RGBtoGray(image)
displayImages()
cv2.imshow('Gray Level Image ',image)
cv2.waitKey(0) 

			
			
	
