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


dataset_path = ["DanaHallWay1", "DanaHallWay2", "DanaOffice"]

for path in dataset_path:
    print(path + '/')

    # Open each image in the specified folder and save them in imgset
    imgset = []
    for serial_number in os.listdir(path + '/'):
        imgset.append(cv2.imread(path + '/' + serial_number))

    w, h = imgset[0].shape[1], imgset[0].shape[0]

    grayscale = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in imgset]

    for frame in grayscale:
        ret = Harris(frame, 3, 3, 0.04)
        corners = non_max_suppression(ret, 5)

        output = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for corner in corners:
            cv2.circle(output, (corner[0], corner[1]), 2, (255, 0, 0))

        plt.imshow(output, cmap='gray')
        plt.show()

