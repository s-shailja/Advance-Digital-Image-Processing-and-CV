from cv2 import *
from PIL import Image
import cv2
import numpy as np
import math
import random
cos=np.zeros((8,8))
C=np.zeros(8)
#Calculate the DCT cosine terms
def CosineTerms():

	for i in range(8):
		for j in range(8):
			cos[i][j]=np.cos((2 * i + 1) * j * np.arccos(-1) / 16.0);
		if (i):
			C[i]=1;
		else:
			C[i]=1/np.sqrt(2)

img=cv2.imread('cap.bmp',0)
img=np.float32(img)
dct=np.zeros((img.shape[0],img.shape[1]))
#DCT transformation
def DCT(): 
	for r in range(img.shape[0]//8):
		for c in range(img.shape[1]//8):
			for i in range(8):
				for j in range(8):
					su=0
					for x in range(8):
						for y in range(8):
							su+=(img[r*8+x][c*8+y]-128)*cos[x][i]*cos[y][j]
					su*=C[i]*C[j]*0.25
					dct[r*8+i][c*8+j]=su

	print("DCT Tranformed Image")	
	print(dct)

def Quantizer():
	table=[[16, 11, 10, 16, 24, 40, 51, 61],[ 12, 12, 14, 19, 26, 58, 60, 55],[14, 13, 16, 24, 40, 57, 69, 56],[14, 17, 22, 29, 51, 87, 80, 82],[18, 22, 37, 56, 68, 109, 103, 77],[24, 35, 55, 64, 81, 104, 113, 92],[49, 64, 78, 87, 103, 121, 120, 101],[72, 92, 95, 98, 112, 100, 103, 99]]
	for r in range(img.shape[0]//8):
		for c in range(img.shape[1]//8):
			for i in range(8):
				for j in range(8):
					dct[r*8+i][c*8+j]=round(dct[r * 8 + i][c * 8 + j] / table[i][j])
	          			dct[r * 8 + i][c * 8 + j] = dct[r * 8 + i][c * 8 + j] * table[i][j]
	print("Quantized DCT")
	print(dct)
def IDCT():
	image=cv2.imread('cap.bmp',-1)
	for r in range(img.shape[0]//8):
		for c in range(img.shape[1]//8):
			for i in range(8):
				for j in range(8):
					su=0
					for x in range(8):
						for y in range(8):
							su+=C[x] * C[y] * dct[r * 8 + x][c * 8 + y] * cos[i][x] * cos[j][y]
					su*=0.25
					su+=128
					image[r*8+i][c*8+j]=su
	print("Decompressed Image")
	print(image)
	cv2.imshow('Decompressed Image',image)	
	cv2.imshow('Original Image',cv2.imread('cap.bmp',0))
	cv2.waitKey(0)
CosineTerms()
DCT()
cv2.imshow('DCT Transformed Image',dct)
Quantizer()
IDCT()



