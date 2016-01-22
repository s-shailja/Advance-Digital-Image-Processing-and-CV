import numpy as np
import random
import cv2
import math

image=cv2.imread('cap.bmp',0)
#Q(3)Add salt and pepper noise where a pixel is turned into  dark (0) or white (255) with a probability of (p/2) for each  (p to be provided as an input parameter) to 'cap.bmp' image 
def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    prob=prob/2
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
	    #print(rdn)
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
print('Enter the value of p')
x=float(input())
cv2.waitKey(0)
noise_img = sp_noise(image,x) 
cv2.imwrite('sp_noise.jpg', noise_img)
cv2.imshow('Noise Image', noise_img)
cv2.waitKey(0)
#Q(3) Median Filter
def MedianFilter(source):    
    final = source[:]
    for y in range(1,source.shape[0]-1):
    	for x in range(1,source.shape[1]-1):
            final[y,x]=source[y,x]
    cv2.imshow('Source_Picture', source) #Show the image
    members=[source[0,0]]*9
    #print(members)
    for y in range(1,source.shape[0]-1):
    	for x in range(1,source.shape[1]-1):
            members[0] = source[y-1,x-1]
	    #print(members[0])
            members[1] = source[y,x-1]
            members[2] = source[y+1,x-1]
            members[3] = source[y-1,x]
            members[4] = source[y,x]
            members[5] = source[y+1,x]
            members[6] = source[y-1,x+1]
            members[7] = source[y,x+1]
            members[8] = source[y+1,x+1]
	    members.sort()
            final[y,x]=members[4]
    source=cv2.imread('cap.bmp',0)
    
    cv2.imshow('Denoised Image (Median Filter) ', final) #Show the image
    cv2.waitKey()
MedianFilter(noise_img)
cv2.waitKey(0)

#Mean Filter
def MeanFilter(source):    
    final = source[:]
    for y in range(1,source.shape[0]-1):
    	for x in range(1,source.shape[1]-1):
            final[y,x]=source[y,x]
    cv2.imshow('Source_Picture', source) #Show the image
    members=[source[0,0]]*9
    #print(members)
    for y in range(1,source.shape[0]-1):
    	for x in range(1,source.shape[1]-1):
            members[0] = source[y-1,x-1]
	    #print(members[0])
            members[1] = source[y,x-1]
            members[2] = source[y+1,x-1]
            members[3] = source[y-1,x]
            members[4] = source[y,x]
            members[5] = source[y+1,x]
            members[6] = source[y-1,x+1]
            members[7] = source[y,x+1]
            members[8] = source[y+1,x+1]	    
            final[y,x]=sum(members)/9
    source=cv2.imread('cap.bmp',0)
    cv2.imshow('Denoised Image (Mean Filter)', final) #Show the image
    cv2.waitKey()
img=cv2.imread('sp_noise.jpg',0)
MeanFilter(img)
cv2.waitKey(0)

