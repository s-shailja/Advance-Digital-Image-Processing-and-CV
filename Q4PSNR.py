import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt
img=cv2.imread('cap.bmp',0)
I1=cv2.imread('cap.bmp',0)
li_med=[]
li_mean=[]
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
def PSNR(I2):
	
        s1 = cv2.absdiff(I1, I2)
        s1 = np.float32(s1)
        s1 = cv2.multiply(s1, s1)

        s = cv2.sumElems(s1)
        sse = s[0] + s[1] + s[2]

        if (sse <= 1e-10):
             return 0
        else:
             mse = sse/(len(I1.shape) * I1.shape[0]*I1.shape[1])
             psnr = 10*math.log((255*255)/mse, 10)		
             return psnr

def MedianFilter(source):    
    final = source[:]
    for y in range(1,source.shape[0]-1):
    	for x in range(1,source.shape[1]-1):
            final[y,x]=source[y,x]
    
    members=[source[0,0]]*9
    
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
    li_med.append(PSNR(final))
    
    
def MeanFilter(source):    
    final = source[:]
    for y in range(1,source.shape[0]-1):
    	for x in range(1,source.shape[1]-1):
            final[y,x]=source[y,x]
    #cv2.imshow('Source_Picture', source) #Show the image
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
    li_mean.append(PSNR(final))
    	
for i in range(1,10,1):
    im=sp_noise(img,i*0.1)
    MedianFilter(im)
    im=sp_noise(img,i*0.1)
    MeanFilter(im)
t=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]	   
plt.plot(t,li_mean,'g--',t,li_med,'r--')
plt.xlabel('Probability')
plt.ylabel('PSNR value')
plt.title('PSNR value vs p')

plt.show()
print(li_mean)
print(li_med)
