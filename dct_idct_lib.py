import cv2
import numpy as np

std_luminance_quant_tbl = [
  [16,  11,  10,  16,  24,  40,  51,  61],
  [12,  12,  14,  19,  26,  58,  60,  55],
  [14,  13,  16,  24,  40,  57,  69,  56],
  [14,  17,  22,  29,  51,  87,  80,  62],
  [18,  22,  37,  56,  68, 109, 103,  77],
  [24,  35,  55,  64,  81, 104, 113,  92],
  [49,  64,  78,  87, 103, 121, 120, 101],
  [72,  92,  95,  98, 112, 100, 103,  99]
]

def DCT(imgfile):
    
    img = cv2.imread(imgfile, 0)
    wndName = "original grayscale"
    cv2.namedWindow(wndName, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(wndName, img)

    iHeight, iWidth = img.shape[:2]
    print "size:", iWidth, "x", iHeight

    # set size to multiply of 8
    if (iWidth % 8) != 0:
        filler = img[:,iWidth-1:]
        #print "width filler size", filler.shape
        for i in range(8 - (iWidth % 8)):
            img = np.append(img, filler, 1)

    if (iHeight % 8) != 0:
        filler = img[iHeight-1:,:]
        #print "height filler size", filler.shape
        for i in range(8 - (iHeight % 8)):
            img = np.append(img, filler, 0)

    iHeight, iWidth = img.shape[:2]
    print "new size:", iWidth, "x", iHeight

    # array as storage for DCT + quant.result
    img2 = np.empty(shape=(iHeight, iWidth))

    # FORWARD ----------------
    # do calc. for each 8x8 non-overlapping blocks
    for startY in range(0, iHeight, 8):
        for startX in range(0, iWidth, 8):
            block = img[startY:startY+8, startX:startX+8]

            # apply DCT for a block
            blockf = np.float32(block)     # float conversion
            
            dst = cv2.dct(blockf)          # dct
                
            # quantization of the DCT coefficients
            blockq = np.around(np.divide(dst, std_luminance_quant_tbl))
            blockq = np.multiply(blockq, std_luminance_quant_tbl)

            # store the result
            for y in range(8):
                for x in range(8):
                    img2[startY+y, startX+x] = blockq[y, x]

    
    # INVERSE ----------------
    for startY in range(0, iHeight, 8):
        for startX in range(0, iWidth, 8):
            block = img2[startY:startY+8, startX:startX+8]
            
            blockf = np.float32(block)     # float conversion
            dst = cv2.idct(blockf)         # inverse dct
            np.place(dst, dst>255.0, 255.0)     # saturation
            np.place(dst, dst<0.0, 0.0)         # grounding 
            block = np.uint8(np.around(dst))

            # store the results
            for y in range(8):
                for x in range(8):
                    img[startY+y, startX+x] = block[y, x]


    wndName1 = "DCT + quantization + inverseDCT"
    cv2.namedWindow(wndName1, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(wndName1, img)
    cv2.imshow('DCT transformed Image',img2)
    cv2.waitKey(0) 
    cv2.destroyWindow('DCT transformed Image')    
    cv2.destroyWindow(wndName1)
    cv2.destroyWindow(wndName)
if __name__ == "__main__":
    DCT("cap.bmp")
