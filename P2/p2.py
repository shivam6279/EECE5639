import numpy as np 
from matplotlib import pyplot as plt 
import math 
import cv2
import os 
import gc

def Harris(img, window_size=3, sobel_size=3, k=0.04, step_size=2):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape[0], img.shape[1]

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_size)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_size)

    Ixx = sobelx ** 2
    Iyy = sobely ** 2
    Ixy = sobelx * sobely
    R = np.zeros_like(img, dtype=np.float64)

    offset = np.floor(window_size / 2).astype(np.int8)
    for y in range(offset, h-offset, step_size):
        for x in range(offset, w-offset, step_size):
            Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
            Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])
            r = (Sxx * Syy) - (Sxy ** 2) - k * (Sxx + Syy) ** 2
            if r < 0:
                r = 0
            R[y][x] = r

    R /= np.max(R)
    R *= 255.0
    # _, R = cv2.threshold(R, 0.01 * np.max(R), 255, 0)
    return R


def non_max_suppression(img, window_size=5, thresh=2.5):
    h, w = img.shape[0], img.shape[1]

    offset = np.floor(window_size / 2).astype(np.int8)

    for y in range(h):
        for x in range(w):
            if x <= offset or x >= (w - offset) or y <= offset or y >= (h - offset):
                img[y][x] = 0

    index = np.zeros((h, w, 3), dtype=np.uint16)
    for y in range(h):
        for x in range(w):
            index[y][x] = (img[y][x], x, y)

    corners = []

    for y in range(offset, h - offset, offset):
        for x in range(offset, w - offset, offset):
            ret = mean_shift_converge(index, (x, y), window_size, thresh)
            if ret:
                corners.append(ret)

    list(set(corners))
    return corners

def mean_shift_converge(index, point, window_size=5, thresh=2.5):
    x, y = point
    offset = np.floor(window_size / 2).astype(np.int8)

    window = index[y - offset:y + 1 + offset, x - offset:x + 1 + offset]
    if np.max(window[:, :, 0]) > thresh:
        window = np.reshape(window, (window_size ** 2, 3))
        window[::-1] = window[window[:, 0].argsort()]
        if window[0][1] == x and window[0][2] == y:
            return x, y
        else:
            return mean_shift_converge(index, (window[0][1], window[0][2]), window_size, thresh)
    else:
        return None


def NCC(imgset, corners):

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

                    if (value > 0.9):
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
    
    ret1 = Harris(grayscale[0], 3, 3, 0.04)
    cor1 = non_max_suppression(ret1, 5)

    ret2 = Harris(grayscale[1], 3, 3, 0.04)
    cor2 = non_max_suppression(ret2, 5)
    
    corners = [cor1, cor2]    

    corres = NCC(grayscale,corners)
    print (corres)
