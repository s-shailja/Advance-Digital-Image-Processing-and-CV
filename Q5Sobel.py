import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import matplotlib.pyplot as plt
img=cv2.imread('lego.tif',0)
hist=[0]*256
p=[0]*256
sigma_b=[0]*256
for y in range(0,img.shape[0]):
    for x in range(0,img.shape[1]):
	hist[img[y][x]]+=1
for i in range(256):
	p[i]=hist[i]/(img.shape[0]*img.shape[1]*1.0)
for t in range (1,256):
   q_L = sum(p[1 : t])
   q_H = sum(p[t + 1 : ])
   miu_L=0
   miu_H=0
   if q_L!=0 and q_H!=0 :
   	for i in range(t):	
   		miu_L += i*p[i] / (q_L*1.0)
   	for i in range(t,256): 
   		miu_H  += i*p[i]  / (q_H*1.0)
   sigma_b[t] = q_L * q_H * (miu_L - miu_H)**2
max=0
ans=0
for i in range(256):
	if max<sigma_b[i]:
		ans=i
		max=sigma_b[i] 
th=ans
tl=ans//2

sobelx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobely=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
gradx=signal.convolve2d(img,sobelx,boundary='symm', mode='same')
grady=signal.convolve2d(img,sobely,boundary='symm', mode='same')
g= img[:]

for y in range(0,gradx.shape[0]):
	for x in range(0,gradx.shape[1]):
		g[y,x]=(gradx[y,x]**2+grady[y,x]**2)**0.5


plt.subplot(2,2,1),plt.imshow(gradx,cmap = 'gray')
plt.title('X-Gradient'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(grady,cmap = 'gray')
plt.title('Y-Gradient'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(g,cmap = 'gray')
plt.title('Gradient Magnitude'), plt.xticks([]), plt.yticks([])

for y in range(0,gradx.shape[0]):
	for x in range(0,gradx.shape[1]):
		if g[y][x]>th:
			g[y][x]=255
		elif g[y][x]<tl:
			g[y][x]=0
plt.subplot(2,2,4),plt.imshow(g,cmap = 'gray')
plt.title('Thresholded Edges'), plt.xticks([]), plt.yticks([])
plt.show()



