import numpy as np 
from matplotlib import pyplot as plt 
import math 
import cv2
import os 
import gc

def Harris(img, window_size=3, k=0.04):
    h, w = img.shape[0], img.shape[1]

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

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


def NCC(imgset, corners):
    h, w = imgset[0].shape[0], imgset[0].shape[1]
    h1, w1 = imgset[1].shape[0], imgset[1].shape[1]
    res = []
    for i in corners[0]:
        if (i[1] != 0 and i[0] != 0 and i[1] != 339 and  i[0] != 511  ):

            f = imgset[0][i[1]-1:i[1]+2, i[0]-1:i[0]+2]
            
            avg = np.sum(f)/9
            for row in range(0,3):
                for col in range (0,3): 
                    f[row][col] -= avg
        
            stdv = math.sqrt(np.sum(f**2))
            for row in range(0,3):
                for col in range (0,3): 
                    f[row][col] /= stdv

            for j in corners[1]:
                if (j[1] != 0 and j[0] != 0 and j[1] != 339 and  j[0] != 511  ):
                    corres = []
                    g = imgset[1][j[1]-1:j[1]+2, j[0]-1:j[0]+2]
                 
                    avg = np.sum(g)/9
                    for row in range(0,3):
                        for col in range (0,3): 
                            g[row][col] -= avg
        
                    stdv = math.sqrt(np.sum(g**2))
                    for row in range(0,3):
                        for col in range (0,3): 
                            g[row][col] /= stdv
                    
                    value = np.sum(f*g)

                    if (value > 0.5):
                        corres.append(i)
                        corres.append(j)
                        res.append(corres)
    
    return(res)


dataset_path = ["DanaHallWay1"]#, "DanaHallWay2", "DanaOffice"]

for path in dataset_path:

    imgset = []
    for serial_number in os.listdir(path + "/"):
        if (len(imgset) < 2):
            imgset.append(cv2.imread(path + "/" + serial_number))

    grayscale = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)for img in imgset]
    grayscale = [np.float32(img) for img in grayscale]

    cor = [cv2.cornerHarris(gray,2,3,0.04) for gray in grayscale]
    cor = [cv2.dilate(img,None) for img in cor]

    ret, cor[0] = cv2.threshold(cor[0],0.001*cor[0].max(),255,0)
    ret, cor[1] = cv2.threshold(cor[1],0.01*cor[1].max(),255,0)

    cor[0] = np.uint8(cor[0])
    cor[1] = np.uint8(cor[1])

    _, _, _, centroids0 = cv2.connectedComponentsWithStats(cor[0])
    _, _, _, centroids1 = cv2.connectedComponentsWithStats(cor[1])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners0 = cv2.cornerSubPix(grayscale[0],np.float32(centroids0),(5,5),(-1,-1),criteria)
    corners1 = cv2.cornerSubPix(grayscale[1],np.float32(centroids1),(5,5),(-1,-1),criteria)
    corners0 = np.int0(corners0)
    corners1 = np.int0(corners1)
    corners = [corners0, corners1]
    
    corres = NCC(grayscale,corners)
    print (corres[2][0])
