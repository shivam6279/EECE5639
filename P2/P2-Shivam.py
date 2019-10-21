import numpy as np
from matplotlib import pyplot as plt
import matplotlib.path as mplPath
import math
import cv2
import os
import gc
import time


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


def remove_values_from_list(the_list, val):
    return [value for value in the_list if value[1] != val]


def filter_NCC(NCC_list, score_thresh=0.6, dif_thresh=0.1, height_thresh=20):
    NCC_list = [NCC_list[i] for i in range(len(NCC_list)) if NCC_list[i][0] > score_thresh]
    NCC_list.sort(key=lambda x: x[0], reverse=True)

    if dif_thresh > 0.01:
        while True:
            flag = 0
            for i in range(len(NCC_list)):
                for j in range(i+1, len(NCC_list)):
                    if NCC_list[i][1] == NCC_list[j][1] and NCC_list[i][0] - NCC_list[j][0] <= dif_thresh:
                        NCC_list = remove_values_from_list(NCC_list, NCC_list[i][1])
                        flag = 1
                        break
                if flag:
                    break
            if flag == 0:
                break

    return NCC_list


def est_homography(points1, points2):
    (x1, y1) = np.array(points1[0], dtype=np.float64)
    (x2, y2) = np.array(points1[1], dtype=np.float64)
    (x3, y3) = np.array(points1[2], dtype=np.float64)
    (x4, y4) = np.array(points1[3], dtype=np.float64)

    (x1_, y1_) = np.array(points2[0], dtype=np.float64)
    (x2_, y2_) = np.array(points2[1], dtype=np.float64)
    (x3_, y3_) = np.array(points2[2], dtype=np.float64)
    (x4_, y4_) = np.array(points2[3], dtype=np.float64)

    # Assume H33 = 1
    A = np.array([[x1, y1, 1, 0,  0,  0, -x1*x1_, -y1*x1_],
                  [0,  0,  0, x1, y1, 1, -x1*y1_, -y1*y1_],
                  [x2, y2, 1, 0,  0,  0, -x2*x2_, -y2*x2_],
                  [0,  0,  0, x2, y2, 1, -x2*y2_, -y2*y2_],
                  [x3, y3, 1, 0,  0,  0, -x3*x3_, -y3*x3_],
                  [0,  0,  0, x3, y3, 1, -x3*y3_, -y3*y3_],
                  [x4, y4, 1, 0,  0,  0, -x4*x4_, -y4*x4_],
                  [0,  0,  0, x4, y4, 1, -x4*y4_, -y4*y4_]], dtype=np.float64)

    B = np.transpose(np.array([x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_], dtype=np.float64))

    h = np.linalg.solve(A, B)

    H = np.array([[h[0], h[1], h[2]],
                  [h[3], h[4], h[5]],
                  [h[6], h[7],  1]])

    np_points1 = np.array(points1)
    np_points2 = np.array(points2)

    H, _ = cv2.findHomography(np_points2, np_points1, method=cv2.RANSAC)

    return H


def get_point_from_homography(p, H):
    den = H[2][0]*p[0] + H[2][1]*p[1] + 1
    x = (H[0][0]*p[0] + H[0][1]*p[1] + H[0][2]) / den
    y = (H[1][0]*p[0] + H[1][1]*p[1] + H[1][2]) / den
    return np.array([x, y], dtype=np.int16)


def main():
    dataset_path = ["DanaHallWay1", "DanaHallWay2", "DanaOffice"]

    for path in dataset_path:
        print(path)

        # Open each image in the specified folder and save them in imgset
        imgset = []
        for serial_number in os.listdir(path + '/'):
            imgset.append(cv2.imread(path + '/' + serial_number))

        w, h = imgset[0].shape[1], imgset[0].shape[0]

        current_img = imgset[0]
        x_shift, y_shift = 0, 0

        for img_no in range(1, len(imgset)):
            w_cur, h_cur = current_img.shape[1], current_img.shape[0]

            corners = []

            ret = Harris(current_img, 3, 3, 0.04)
            centroids = non_max_suppression(ret, 9)
            corners.append(centroids)

            ret = Harris(imgset[img_no], 3, 3, 0.04)
            centroids = non_max_suppression(ret, 9)
            corners.append(centroids)

            NCC_score = []
            for i in range(len(corners[0])):
                for j in range(len(corners[1])):
                    score = NCC(imgset[0], imgset[1], corners[0][i], corners[1][j], 25)
                    NCC_score.append((score, corners[0][i], corners[1][j]))

            # Filters the NCC list based on some parameters - not very important, just testing things
            NCC_score = filter_NCC(NCC_score)

            # Only send the top 10 NCC corners for the homography calculation - temporary
            p1 = [(NCC_score[i][1][0], NCC_score[i][1][1]) for i in range(min(15, len(NCC_score)))]
            p2 = [(NCC_score[i][2][0], NCC_score[i][2][1]) for i in range(min(15, len(NCC_score)))]

            # Get homography
            H = est_homography(p1, p2)

            out1 = current_img.copy()
            for corner in corners[0]:
                cv2.circle(out1, (corner[0], corner[1]), 2, (0, 0, 255))
            out2 = np.zeros((h_cur, w, 3), dtype=np.uint8)
            out2[y_shift:h + y_shift, 0:w] = imgset[img_no]
            for corner in corners[1]:
                cv2.circle(out2, (corner[0], corner[1] + y_shift), 2, (0, 0, 255))
            combine = np.concatenate((out1, out2), axis=1)
            for i in range(min(8, len(NCC_score))):
                cv2.line(combine, (NCC_score[i][1][0], NCC_score[i][1][1]), (NCC_score[i][2][0] + w_cur, NCC_score[i][2][1] + y_shift), (255, 0, 0), 1)
            plt.imshow(cv2.cvtColor(combine, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

            # Calculate the bounding edges after warping the second image with H
            # c1-c4 are the new corners of the image after warping
            c1 = get_point_from_homography((0, 0), H)
            c2 = get_point_from_homography((w-1, 0), H)
            c3 = get_point_from_homography((0, h-1), H)
            c4 = get_point_from_homography((w-1, h-1), H)
            center_new = get_point_from_homography((w/2, h/2), H)
            # w_new and h_new are the bounding dimensions of the warped image
            w_new = np.max([c4[0]-c1[0], c4[0]-c3[0], c2[0]-c1[0], c2[0]-c3[0]])
            h_new = np.max([c4[1]-c1[1], c4[1]-c2[1], c3[1]-c1[1], c3[1]-c2[1]])
            # x_offset and y_offset are the minimum x, y coordinates of the warped image
            x_offset = np.min([c1[0], c3[0]])
            y_offset = np.min([c1[1], c2[1]])
            # Determine necessary offsets to shift the images so that the take up the entire image
            # This is done because the warp function ignores negative values produced by homography transformation
            # So if the offsets computed earlier are negative, the images need to be shifted
            if y_offset < 0:
                y_shift = -y_offset
            else:
                y_shift = 0

            if x_offset < 0:
                x_shift = -x_offset
            else:
                x_shift = 0

            # Bounding box of the warped image in the combined image coordinates
            merge_offset = 5
            roi = mplPath.Path(np.array([[c1[0] + x_shift + merge_offset, c1[1] + y_shift + merge_offset],
                                         [c2[0] + x_shift - merge_offset, c2[1] + y_shift + merge_offset],
                                         [c4[0] + x_shift - merge_offset, c4[1] + y_shift - merge_offset],
                                         [c3[0] + x_shift + merge_offset, c3[1] + y_shift - merge_offset]]))
            # Offset matrix - since the homography may give negative values after warping
            # The image needs to be shifted accordingly to keep the coordinates positive
            # x_offset = -np.min([c1[0], c3[0]])
            # y_offset = -np.min([c1[1], c2[1]])
            # x_offset, y_offset = NCC_score[0][1] - get_point_from_homography(NCC_score[0][2], H)
            O = np.array([[1, 0, x_shift],
                          [0, 1, y_shift],
                          [0, 0, 1]])
            print(O)
            # Warp the second image based on the Homography and offset matrix with the new image size
            # @ is the notation for matrix multiplication
            output = cv2.warpPerspective(imgset[img_no], O@H, (max(w_new+x_offset, w_cur), max(h_new, h_cur)))
            # output = cv2.addWeighted(out1, 0.5, out2, 0.5, 0.0)
            # output = out.copy()

            plt.figure()
            plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

            print(output.shape)
            print(current_img.shape)
            output[y_shift: h_cur+y_shift, x_shift: w_cur+x_shift] = current_img

            current_img = output

            # r1_den = (y_shift-center_new[1]) ** 2 + (x_shift-center_new[0]) ** 2
            # r2_den = (h/2) ** 2 + (w/2) ** 2
            # for i in range(h):
            #     for j in range(w):
            #         if roi.contains_point((j+x_shift, i+y_shift)):
            #             # r1 = 1 - (i+y_shift-center_new[1]) ** 2 + (j+x_shift-center_new[0]) ** 2 / r1_den
            #             # r2 = 1 - (i-h/2) ** 2 + (j-w/2) ** 2 / r2_den
            #             r1 = 1
            #             r2 = 1
            #             temp = (r1 * output[i+y_shift][j+x_shift].astype(np.float32) + r2 * imgset[0][i][j].astype(np.float32)) / (r1+r2)
            #             output[i+y_shift][j+x_shift] = temp
            #         else:
            #             output[i+y_shift][j+x_shift] = imgset[0][i][j]

            plt.figure()
            plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()


if __name__ == "__main__":
    main()

