import cv2
import numpy as np
from numpy import linalg as LA
from pylab import *
from scipy import signal
from filtertools import gauss_kernel

def gauss_kernel(size, sizey = None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    g = exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()
def compute_harris_response(image):
    """ compute the Harris corner detector response function 
        for each pixel in the image"""

    #derivatives
    sobelx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobely=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    imx=signal.convolve2d(image,sobelx,boundary='symm', mode='same')
    imy=signal.convolve2d(image,sobely,boundary='symm', mode='same')

    #kernel for blurring
    gauss = gauss_kernel(3)
    print(gauss)	
    #compute components of the structure tensor
    Wxx = signal.convolve(imx*imx,gauss, mode='same')
    Wxy = signal.convolve(imx*imy,gauss, mode='same')
    Wyy = signal.convolve(imy*imy,gauss, mode='same')

    #determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    
    return (Wdet/Wtr)

def get_harris_points(harrisim, min_distance=10, threshold=0.1):
    """ return corners from a Harris response image
        min_distance is the minimum nbr of pixels separating 
        corners and image boundary"""

    #find top corner candidates above a threshold
    corner_threshold = max(harrisim.ravel()) * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    #get coordinates of candidates
    candidates = harrisim_t.nonzero()
    coords = [ (candidates[0][c],candidates[1][c]) for c in range(len(candidates[0]))]
    #...and their values
    candidate_values = [harrisim[c[0]][c[1]] for c in coords]

    #sort candidates
    index = argsort(candidate_values)

    #store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_distance:-min_distance,min_distance:-min_distance] = 1
    print('fugihi')
    #select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i][0]][coords[i][1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i][0]-min_distance):(coords[i][0]+min_distance),(coords[i][1]-min_distance):(coords[i][1]+min_distance)] = 0

    return filtered_coords
def plot_harris_points(image, filtered_coords):
    """ plots corners found in image"""
    figure()    
    imshow(image)
    gray()	
    plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')
    axis('off')
    show()    
     
img=cv2.imread('lego.tif',-1)
im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
harrissim=compute_harris_response(im)
filtered_coords=get_harris_points(harrissim,6)
plot_harris_points(im,filtered_coords)
