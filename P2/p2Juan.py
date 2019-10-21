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
            R[y][x] = (Sxx * Syy) - (Sxy ** 2) - k * (Sxx + Syy) ** 2

    R /= np.max(R)
    R *= 255.0
    # _, R = cv2.threshold(R, 0.01 * np.max(R), 255, 0)
    return R


def non_max_suppression(img, window_size=5, thresh=2.5):
    h, w = img.shape[0], img.shape[1]

    offset = np.floor(window_size / 2).astype(np.int8)

    temp = np.zeros_like(img)
    temp[offset:h-offset, offset:w-offset] = img[offset:h-offset, offset:w-offset]
    img = temp
    del temp

    index = np.zeros((h, w, 3), dtype=np.int16)

    index[:, :, 0] = img

    for y in range(h):
        for x in range(w):
            index[y][x][1] = x
            index[y][x][2] = y

    corners = []

    global NMS_index
    NMS_index = np.zeros((h, w))

    for y in range(offset, h - offset, offset):
        for x in range(offset, w - offset, offset):
            ret = mean_shift_converge(index, (x, y), window_size, thresh)
            if ret:
                corners.append(ret)

    corners = list(set(corners))
    corners.sort()
    return corners


def mean_shift_converge(index, point, window_size=5, thresh=2.5):
    x, y = point
    offset = np.floor(window_size / 2).astype(np.int8)

    if NMS_index[y][x] == 1:
        return None

    window = index[y - offset:y + 1 + offset, x - offset:x + 1 + offset]
    if np.max(window[:, :, 0]) > thresh:
        window = np.reshape(window, (window_size ** 2, 3))
        window[::-1] = window[window[:, 0].argsort()]
        if window[0][1] == x and window[0][2] == y:
            NMS_index[y][x] = 1
            return x, y
        else:
            if mean_shift_converge(index, (window[0][1], window[0][2]), window_size, thresh):
                NMS_index[y][x] = 1
                return window[0][1], window[0][2]
            else:
                return None
    else:
        return None


def NCC(img1, img2, p1, p2, window_size=3):
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = img1.shape[0], img1.shape[1]

    offset = np.floor(window_size / 2).astype(np.int8)

    (x1, y1) = p1
    (x2, y2) = p2

    if x1 <= offset or x1 >= (w-offset) or y1 <= offset or y1 >= (h-offset) or\
            x2 <= offset or x2 >= (w - offset) or y2 <= offset or y2 >= (h - offset):
        return 0

    window1 = img1[y1-offset:y1+1+offset, x1-offset:x1+1+offset]
    window1 = window1.astype(np.float64)

    window2 = img2[y2-offset:y2+1+offset, x2-offset:x2+1+offset]
    window2 = window2.astype(np.float64)

    mean1 = np.mean(window1)
    std1 = np.std(window1)

    mean2 = np.mean(window2)
    std2 = np.std(window2)

    s = 0
    for i in range(0, window_size):
        for j in range(0, window_size):
            s += (window1[i][j] - mean1) / std2 * (window2[i][j] - mean2) / std1
    s /= (window_size ** 2)

    return s



def homography(corres):
    p = 0.99
    e = 0.3
    s = len(corres)
    N = np.log(1-p)/(np.log(1-(1-e)**s))
    N = np.int32(N)

    inliners = []
    inlinecount = 0
    resh = []
    
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
      

        h, status= cv2.findHomography(pointsImg1, pointsImg2)
        
        if (status[0][0] != 0):

            counter = 0
            thresh = 3
            temp = []
            for p in corres:
                img2x = (h[0][0] * p[0][0] + h[0][1] * p[0][1] + h[0][2])/(h[2][0] * p[0][0] + h[2][1] * p[0][1] + h[2][2])
                img2y = (h[1][0] * p[0][0] + h[1][1] * p[0][1] + h[1][2])/(h[2][0] * p[0][0] + h[2][1] * p[0][1] + h[2][2])
                img2x = np.ceil(img2x)
                img2y = np.ceil(img2y)

                if ((img2x >= p[1][0] - thresh and img2x <= p[1][0] + thresh)and(img2y >= p[1][1] - thresh and img2y <= p[1][1] + thresh)):
                    counter += 1
                    temp.append(p)

            

            if (counter > inlinecount):
                inliners = temp
                inlinecount = counter
                resh = h
    print(inlinecount)
    return(inliners,resh)

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
    NCC_points = []

    for i in range(len(corners[0])):
        for j in range(len(corners[1])):
            score = NCC(imgset[0], imgset[1], corners[0][i], corners[1][j], 25)
            if (score  > 0.9):
                NCC_points.append((corners[0][i], corners[1][j]))
    print (len(NCC_points))
    corres, h = homography(NCC_points)

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
    
