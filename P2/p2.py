import numpy as np 
from matplotlib import pyplot as plt 
import math 
import cv2
import os 
import gc

def Harris(img, window_size=3, sobel_size=3, k=0.04):
    h, w = img.shape[0], img.shape[1]

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_size)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_size)

    Ixx = sobelx ** 2
    Iyy = sobely ** 2
    Ixy = sobelx * sobely
    R = np.zeros_like(img, dtype=np.float64)

    offset = np.floor(window_size / 2).astype(np.int8)
    for y in range(offset, h-offset):
        for x in range(offset, w-offset):
            Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
            Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])
            det = (Sxx * Syy) - (Sxy ** 2)
            trace2 = (Sxx + Syy) ** 2
            R[y][x] = det - k*trace2
        
    return R



dataset_path = ["DanaHallWay1"]#, "DanaHallWay2", "DanaOffice"]


for path in dataset_path:

    imgset = []
    for serial_number in os.listdir(path + "/"):
        if (len(imgset) < 2):
            imgset.append(cv2.imread(path + "/" + serial_number))

    grayscale = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)for img in imgset]
    grayscale = np.float32(grayscale)

    dst = [cv2.cornerHarris(img,2,3,0.4) for img in grayscale]
    
    for img in dst:
        cv2.imshow("dst", img)

        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
