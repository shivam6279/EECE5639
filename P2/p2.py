import numpy as np 
from matplotlib import pyplot as plt 
import math 
import cv2
import os 
import gc

def Harris(img, window_size=3, sobel_size=3, k=0.04):
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
    for y in range(offset, h-offset):
        for x in range(offset, w-offset):
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


def NCC(imgset, corners, windowSize = 3):

    offset = np.floor(windowSize / 2).astype(np.int8)
    res = []
    for i in corners[0]:
        if (i[1] != 0 and i[0] != 0 and i[1] <= 339 - offset and  i[0] <= 511 - offset  ):

            f = imgset[0][i[1]-offset:i[1]+1+offset, i[0]-offset:i[0]+1+offset]
        
            avg = np.sum(f)/9
            for row in range(0,windowSize):
                for col in range (0,windowSize): 
                    f[row][col] -= avg
        
            stdv = math.sqrt(np.sum(f**2))
            for row in range(0,windowSize):
                for col in range (0,windowSize): 
                    f[row][col] /= stdv

            for j in corners[1]:
                if (j[1] != 0 and j[0] != 0 and j[1] <= 339-offset and  j[0] <= 511 - offset  ):
                    corres = []
                    g = imgset[1][j[1]-offset:j[1]+1+offset, j[0]-offset:j[0]+1+offset]
                 
                    avg = np.sum(g)/9
                    for row in range(0,windowSize):
                        for col in range (0,windowSize): 
                            g[row][col] -= avg
        
                    stdv = math.sqrt(np.sum(g**2))
                    for row in range(0,windowSize):
                        for col in range (0,windowSize): 
                            g[row][col] /= stdv
                    
                    value = np.sum(f*g)

                    if (value > 0.99):
                        corres.append(i)
                        corres.append(j)
                        res.append(corres)
    
    return(res)


def homography(corres):
    p = 0.9
    e = 0.1
    s = len(corres)
    N = np.log(1-p)/(np.log(1-(1-e)**s))
    inliners = []
    inlinecount = 0
    N = np.int32(N)
    for x in range(0,N):
        pointsImg1 = []
        pointsImg2 = []
        c = 0
        while (c < 4): 

            r = np.random.randint(low = 0, high = len(corres))
            if (corres[r][0] not in pointsImg1):
                 
                pointsImg1.append(corres[r][0])
                pointsImg2.append(corres[r][1])
                c +=1

        pointsImg1 = np.array(pointsImg1, ndmin = 2)
        pointsImg2 = np.array(pointsImg2, ndmin = 2)

        h, _= cv2.findHomography(pointsImg1, pointsImg2)

        counter = 0
        thresh = 20
        temp = []
        for p in corres:
            img2x = (h[0][0] * p[0][0] + h[0][1] * p[0][1] + h[0][2])/(h[2][0] * p[0][0] + h[2][1] * p[0][1] + h[2][2])
            img2y = ((h[1][0] * p[0][0] + h[1][1] * p[0][1] + h[1][2]))/(h[2][0] * p[0][0] + h[2][1] * p[0][1] + h[2][2])
            img2x = np.ceil(img2x)
            img2y = np.ceil(img2y)

            if ((img2x >= p[0][0] - thresh and img2x <= p[0][0] + thresh)and(img2y >= p[0][1] - thresh and img2y <= p[0][1] + thresh)):
                counter += 1
                temp.append(p)

        

        if (counter > inlinecount):
            inliners = temp
            inlinecount = counter
    print(inlinecount)

dataset_path = ["DanaHallWay1"]#, "DanaHallWay2", "DanaOffice"]

for path in dataset_path:

    imgset = []
    for serial_number in os.listdir(path + "/"):
        if (len(imgset) < 2):
            imgset.append(cv2.imread(path + "/" + serial_number))

    w, h = imgset[0].shape[1], imgset[0].shape[0]
    grayscale = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)for img in imgset]
    grayscale = [np.float32(img) for img in grayscale]
    
    ret1 = Harris(grayscale[0], 3, 3, 0.04)
    cor1 = non_max_suppression(ret1, 5)

    ret2 = Harris(grayscale[1], 3, 3, 0.04)
    cor2 = non_max_suppression(ret2, 5)
    
    corners = [cor1, cor2]    

    corres = NCC(grayscale,corners)

    out1 = imgset[0].copy()
    for corner in corres:
        cv2.circle(out1,(corner[0][0],corner[0][1]),2,(0,0,255))

    out2 = imgset[1].copy()
    for corner in corres:
        cv2.circle(out2,(corner[1][0],corner[1][1]),2,(0,0,255))
    
    combine = np.concatenate((out1,out2),axis=1)

    for i in corres:
        cv2.line(combine, (i[0][0], i[0][1]), (i[1][0] + w, i[1][1]),(255,0,0),1)

    plt.imshow(combine, cmap='gray')
    plt.show()
    #homography(corres)
    
