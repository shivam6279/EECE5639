import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
import os
import gc
import time
import colorsys


def Harris(img, window_size=3, sobel_size=3, k=0.04, step_size=1):
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
    for y in range(offset, h - offset, step_size):
        for x in range(offset, w - offset, step_size):
            Sxx = np.sum(Ixx[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Syy = np.sum(Iyy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sxy = np.sum(Ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
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

    temp = np.zeros_like(img)
    temp[offset:h - offset, offset:w - offset] = img[offset:h - offset, offset:w - offset]
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


def NCC(img1, img2, c1, c2, window_size=3):
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = img1.shape[0], img1.shape[1]

    offset = np.floor(window_size / 2).astype(np.int8)

    ncc = []
    for p1 in c1:
        for p2 in c2:
            (x1, y1) = p1
            (x2, y2) = p2

            if x1 <= offset or x1 >= (w - offset) or y1 <= offset or y1 >= (h - offset) or \
                    x2 <= offset or x2 >= (w - offset) or y2 <= offset or y2 >= (h - offset):
                continue

            window1 = img1[y1 - offset:y1 + 1 + offset, x1 - offset:x1 + 1 + offset]
            window1 = window1.astype(np.float64)

            window2 = img2[y2 - offset:y2 + 1 + offset, x2 - offset:x2 + 1 + offset]
            window2 = window2.astype(np.float64)

            mean1 = np.mean(window1)
            std1 = np.std(window1)

            mean2 = np.mean(window2)
            std2 = np.std(window2)

            s = 0
            for i in range(0, window_size):
                for j in range(0, window_size):
                    s += (window1[i][j] - mean1) / std1 * (window2[i][j] - mean2) / std2
            s /= (window_size ** 2)

            ncc.append((s, p1, p2))

    ncc.sort(key=lambda x: x[0], reverse=True)
    return ncc


def remove_values_from_list(the_list, val):
    return [value for value in the_list if value[1] != val]


def filter_NCC(NCC_list, len_thresh=0, score_thresh=0):
    NCC_list.sort(key=lambda x: x[0], reverse=True)

    if score_thresh > 0:
        NCC_list = [NCC_list[i] for i in range(len(NCC_list)) if NCC_list[i][0] > score_thresh]

    if 0 < len_thresh <= len(NCC_list):
        NCC_list = NCC_list[:len_thresh]

    return NCC_list


def RANSAC(corres, max_N=10000):
    p = 0.99
    e = 0.6
    s = len(corres)
    N = np.log(1 - p) / (np.log(1 - (1 - e) ** s))
    N = np.int32(N)

    if N > max_N:
        N = max_N

    print("N =", N)

    inliers = []
    inlinecount = 0
    resF = []

    for x in range(0, N):
        pointsImg1 = []
        pointsImg2 = []
        c = 0

        while c < 8:

            r = np.random.randint(low=0, high=len(corres))
            if corres[r][0] not in pointsImg1:
                if corres[r][1] not in pointsImg2:
                    pointsImg1.append(corres[r][0])
                    pointsImg2.append(corres[r][1])
                    c += 1

        F = fundMat8Points(pointsImg1, pointsImg2)

        counter = 0
        thresh = 0.001
        temp = []

        for p in corres:
            points = np.array(p)
            pl = np.append(points[0], [1])
            pr = np.append(points[1], [1])
            plT = [[pl[0]], [pl[1]], [pl[2]]]
            res = pr @ F @ plT

            if -thresh < res < thresh:
                counter += 1
                temp.append(p)

        if counter > inlinecount:
            inlinecount = counter
            inliers = temp
            resF = F

        if counter == len(corres):
            x = N

    if counter == 0:
        print("ERROR!!!")

    print(len(corres), inlinecount)

    return np.array(inliers), resF


def fundMat8Points(pImg1, pImg2):
    A = []

    pImg1 = np.array(pImg1)
    pImg2 = np.array(pImg2)
    ones = np.ones((8, 1))
    pImg1 = np.hstack((pImg1, ones))
    pImg2 = np.hstack((pImg2, ones))

    ux1 = np.mean(pImg1[:, 0])
    uy1 = np.mean(pImg1[:, 1])
    ux2 = np.mean(pImg2[:, 0])
    uy2 = np.mean(pImg2[:, 1])

    T1 = [[1, 0, -ux1], [0, 1, -uy1], [0, 0, 1]]
    T2 = [[1, 0, -ux2], [0, 1, -uy2], [0, 0, 1]]

    for i in range(0, 8):
        pImg1[i] = pImg1[i] @ T1
        pImg2[i] = pImg2[i] @ T2

    sdvx1 = np.std(pImg1[:, 0])
    sdvy1 = np.std(pImg1[:, 1])
    sdvx2 = np.std(pImg2[:, 0])
    sdvy2 = np.std(pImg2[:, 1])

    T1 = np.array([[1 / sdvx1, 0, 0], [0, 1 / sdvy1, 0], [0, 0, 1]])
    T2 = np.array([[1 / sdvx2, 0, 0], [0, 1 / sdvy2, 0], [0, 0, 1]])

    for i in range(0, 8):
        pImg1[i] = T1 @ pImg1[i]
        pImg2[i] = T2 @ pImg2[i]

    T1 = [[1 / sdvx1, 0, -ux1], [0, 1 / sdvy1, -uy1], [0, 0, 1]]
    T2 = [[1 / sdvx2, 0, -ux2], [0, 1 / sdvy2, -uy2], [0, 0, 1]]

    for i in range(0, 8):
        A.append([pImg1[i][0] * pImg2[i][0],
                  pImg1[i][0] * pImg2[i][1],
                  pImg1[i][0],
                  pImg1[i][1] * pImg2[i][0],
                  pImg1[i][1] * pImg2[i][1],
                  pImg1[i][1],
                  pImg2[i][0],
                  pImg2[i][1],
                  1])

    _, _, v = np.linalg.svd(A)
    vT = np.transpose(v)
    F = vT[:, 8]
    F = np.reshape(F, (3, 3))

    u, d, v = np.linalg.svd(F)
    d[2] = 0
    F = (u * d[..., None, :]) @ v

    T1 = np.array(T1)
    T2 = np.array(T2)
    F = np.transpose(T2) @ F @ T1
    return F


def display_corner_pairings(img1, img2, lines, y1_shift=0, y2_shift=0):
    w1, h1 = img1.shape[1], img1.shape[0]
    w2, h2 = img2.shape[1], img2.shape[0]

    w_max, h_max = np.max((w1, w2)), np.max((h1, h2))

    out1 = np.zeros((h_max, w_max, 3), dtype=np.uint8)
    out1[y1_shift:h1 + y1_shift, 0:w1] = img1
    out2 = np.zeros((h_max, w_max, 3), dtype=np.uint8)
    out2[y2_shift:h2 + y2_shift, 0:w2] = img2
    combine = np.concatenate((out1, out2), axis=1)

    for p in lines:
        cv2.circle(combine, (p[0][0], p[0][1] + y1_shift), 2, (0, 0, 255), 1)
        cv2.circle(combine, (p[1][0] + w1, p[1][1] + y2_shift), 2, (0, 0, 255), 1)

        hsv = np.array((np.random.rand(), 1, 1))
        rgb = list((np.array(colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])) * 255.0).astype(np.uint16))
        rgb = [int(rgb[0]), int(rgb[1]), int(rgb[2])]
        cv2.line(combine, (p[0][0], p[0][1] + y1_shift), (p[1][0] + w1, p[1][1] + y2_shift), rgb, 1)

    plt.imshow(combine)
    plt.axis('off')
    plt.show()


def calc_disparity(img1, img2, F, c1, c2, SSD_window_size=7):
    w, h = img1.shape[1], img1.shape[0]

    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    x_d = c1[:, 0] - c2[:, 0]
    y_d = c1[:, 1] - c2[:, 1]

    x_d_avg = np.average(x_d)
    y_d_avg = np.average(y_d)
    x_d_std = np.std(x_d)
    y_d_std = np.std(y_d)
    x_d_min = np.min(x_d)
    x_d_max = np.max(x_d)
    y_d_min = np.min(y_d)
    y_d_max = np.max(y_d)
    x_d_mid = (x_d_max + x_d_min) / 2
    y_d_mid = (y_d_max + y_d_min) / 2

    window_size = (x_d_max - x_d_min)+4
    if window_size % 2 == 0:
        window_size += 1

    offset = np.floor(window_size / 2).astype(np.int8)
    SSD_offset = np.floor(SSD_window_size / 2).astype(np.int8)

    print("x min:", x_d_min, "y min:", y_d_min)
    print("x max:", x_d_max, "y max:", y_d_max)
    print("x std:", x_d_std, "y std:", y_d_std)
    print("x avg:", x_d_avg, "y avg:", y_d_avg)
    print("x mid:", x_d_mid, "y mid:", y_d_mid)
    print("Window size:", window_size)

    x_disparity = np.full((h, w), -1, dtype=np.float32)
    y_disparity = np.full((h, w), -1, dtype=np.float32)
    disparity = np.full((h, w, 2), -1, dtype=np.float32)

    for i in range((offset+SSD_offset), h-(offset+SSD_offset), 1):
        for j in range((offset+SSD_offset), w-(offset+SSD_offset), 1):
            x_new = int(j - x_d_mid)
            y_new = int(i - y_d_mid)
            # print(x_new, y_new)

            if not ((offset+SSD_offset) <= x_new < (w-(offset+SSD_offset)) and (offset+SSD_offset) <= y_new <= (h-(offset+SSD_offset))):
                continue

            p = [(a, b) for b in range(y_new-1, y_new + 2) for a in range(x_new - offset, x_new + 1 + offset)]
            SSD = []

            window1 = img1[i - SSD_offset:i + 1 + SSD_offset, j - SSD_offset:j + 1 + SSD_offset]
            for (px, py) in p:
                window2 = img2[py - SSD_offset:py + 1 + SSD_offset, px - SSD_offset:px + 1 + SSD_offset]
                ssd = np.average((window1-window2) ** 2)
                SSD.append((ssd, (px, py)))

            SSD.sort(key=lambda x: x[0], reverse=False)

            x_disparity[i][j] = j - SSD[0][1][0]
            y_disparity[i][j] = i - SSD[0][1][1]

    # temp_x = [x_disparity[i][j] for i in range(h) for j in range(w) if x_disparity[i][j] != -1]
    # temp_y = [x_disparity[i][j] for i in range(h) for j in range(w) if y_disparity[i][j] != -1]
    #
    # min_x = np.min(x_disparity)  # min(temp_x)
    # max_x = np.max(x_disparity)  # np.max(temp_x)
    #
    # min_y = np.min(y_disparity)  # min(temp_y)
    # max_y = np.max(y_disparity)  # max(temp_y)
    #
    # if max_x != min_x:
    #     x_disparity *= 255.0 / (max_x - min_x)
    # else:
    #     x_disparity *= 255.0 / max_x
    #
    # if max_y != min_y:
    #     y_disparity *= 255.0 / (max_y - min_y)
    # else:
    #     y_disparity *= 255.0 / max_y

    disparity[:, :, 0] = np.sqrt((y_disparity - x_disparity) ** 2)
    disparity[:, :, 1] = (np.arctan2(y_disparity, x_disparity) + np.pi) * 180.0 / np.pi

    x_disparity = np.fabs(x_disparity).astype(np.uint8)
    y_disparity = np.fabs(y_disparity).astype(np.uint8)
    disparity = np.fabs(disparity).astype(np.uint8)
    return x_disparity, y_disparity, disparity


def display_vector(v):
    hsv = np.array((np.random.rand(), 1, 1))
    rgb = list((np.array(colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])) * 255.0).astype(np.uint16))
    rgb = [int(rgb[0]), int(rgb[1]), int(rgb[2])]


def main():
    dataset_path = ["cones", "cast"]

    for path in dataset_path:
        print(path)

        # Open first 2 images in the specified folder and save them in imgset
        imgset = []
        for i in range(2):
            imgset.append(cv2.cvtColor(cv2.imread(path + '/' + os.listdir(path + '/')[i]), cv2.COLOR_BGR2RGB))

        w, h = imgset[0].shape[1], imgset[0].shape[0]

        # Get corners for the first image
        a = time.time()  # Timing Statements
        ret = Harris(imgset[0], 3, 3, 0.04, step_size=2)
        centroids = non_max_suppression(ret, 9)
        corners = [centroids]

        # Get corners for the second image
        ret = Harris(imgset[1], 3, 3, 0.04, step_size=2)
        centroids = non_max_suppression(ret, 9)
        corners.append(centroids)

        out1 = imgset[0].copy()
        out2 = imgset[1].copy()
        combine = np.concatenate((out1, out2), axis=1)
        for p in corners[0]:
            cv2.circle(combine, (p[0], p[1]), 3, (255, 0, 0), 1)
        for p in corners[1]:
            cv2.circle(combine, (p[0] + w, p[1]), 3, (255, 0, 0), 1)

        a = time.time()
        NCC_score = NCC(imgset[0], imgset[1], corners[0], corners[1], 5)
        # Filters the NCC list based on some parameters - sort by score with a threshold, min length
        NCC_score = filter_NCC(NCC_score, len_thresh=20)
        print("NCC: ", time.time() - a)  # Timing Statements

        for p in NCC_score:
            cv2.circle(combine, (p[1][0], p[1][1]), 3, (0, 255, 0), 1)
            cv2.circle(combine, (p[2][0] + w, p[2][1]), 3, (0, 255, 0), 1)

        plt.imshow(combine)
        plt.axis('off')
        plt.show()

        # Display corner pairings before RANSAC
        p1 = [(p[1], p[2]) for p in NCC_score]
        display_corner_pairings(imgset[0], imgset[1], p1)

        inliers, F = RANSAC(p1)

        p1 = np.array([(p[1][0], p[1][1]) for p in NCC_score], dtype=np.float64)
        p2 = np.array([(p[2][0], p[2][1]) for p in NCC_score], dtype=np.float64)
        F, _ = cv2.findFundamentalMat(p1, p2, method=cv2.FM_RANSAC)
        print(F)

        # Display corner pairings after RANSAC
        display_corner_pairings(imgset[0], imgset[1], inliers)

        a = time.time()
        print(imgset[0].shape, imgset[1].shape)
        x_disparity, y_disparity, disparity = calc_disparity(imgset[0], imgset[1], F, inliers[:, 0, :], inliers[:, 1, :])
        print("Disparity", time.time() - a)  # Timing Statements

        cv2.imwrite(path + "/x.jpg", x_disparity)
        cv2.imwrite(path + "/y.jpg", y_disparity)

        plt.imshow(x_disparity, cmap='gray')
        plt.axis('off')
        plt.show()

        plt.imshow(y_disparity, cmap='gray')
        plt.axis('off')
        plt.show()

        plt.imshow(disparity[:, :, 0], cmap='gray')
        plt.axis('off')
        plt.show()


def test():
    for c in range(2):
        if c == 0:
            x_disparity = cv2.imread("cones_x.jpg", 0)
            y_disparity = cv2.imread("cones_y.jpg", 0)
            x_disparity = x_disparity[20:354, 55:429].astype(np.float32)
            y_disparity = y_disparity[20:354, 55:429].astype(np.float32)
        else:
            x_disparity = cv2.imread("cast_x.jpg", 0)
            y_disparity = cv2.imread("cast_y.jpg", 0)
            x_disparity = x_disparity[21:361, 93:554].astype(np.float32)
            y_disparity = y_disparity[21:361, 93:554].astype(np.float32)

        w, h = x_disparity.shape[1], x_disparity.shape[0]

        disparity = np.zeros((h, w, 2), dtype=np.float32)
        disparity[:, :, 0] = np.sqrt((y_disparity - x_disparity) ** 2)
        disparity[:, :, 1] = (np.arctan2(y_disparity, x_disparity) + np.pi) * 180.0 / np.pi

        min_x = np.min(x_disparity)  # min(temp_x)
        max_x = np.max(x_disparity)  # np.max(temp_x)

        min_y = np.min(y_disparity)  # min(temp_y)
        max_y = np.max(y_disparity)  # max(temp_y)

        x_disparity = 255.0 * (x_disparity - min_x) / (max_x - min_x)
        y_disparity = 255.0 * (y_disparity - min_y) / (max_y - min_y)

        x_disparity = np.fabs(x_disparity).astype(np.uint8)
        y_disparity = np.fabs(y_disparity).astype(np.uint8)

        RGB = np.zeros((h, w, 3), dtype=np.float32)

        max_mag = np.max(disparity[:, :, 0])
        min_mag = np.min(disparity[:, :, 0])

        max_ang = np.max(disparity[:, :, 1])
        min_ang = np.min(disparity[:, :, 1])
        for i in range(h):
            for j in range(w):
                h = (disparity[i][j][1] - min_ang) / (max_ang - min_ang) * 0.25 + 0.75
                s = (disparity[i][j][0] - min_mag) / (max_mag - min_mag)
                hsv = np.array((h, s, 1))
                rgb = list((np.array(colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])) * 255.0).astype(np.uint16))
                RGB[i][j] = [int(rgb[0]), int(rgb[1]), int(rgb[2])]

        RGB = RGB.astype(np.uint8)
        print(np.max(RGB[:, :, 0]))

        plt.imshow(x_disparity, cmap='gray')
        plt.axis('off')
        plt.show()

        plt.imshow(y_disparity, cmap='gray')
        plt.axis('off')
        plt.show()

        plt.imshow(RGB)
        plt.axis('off')
        plt.show()

        plt.imshow(disparity[:, :, 1], cmap='gray')
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    # main()
    test()

