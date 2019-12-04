import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
import os
import gc
import scipy.io as sio


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


def generatePatch():
    train_path = "CarTrainImages"

    WINDOW_SIZE = 25
    OFFSET = int(np.floor(WINDOW_SIZE/2))

    patch = []
    displacement = []

    i = 0
    for serial_number in os.listdir(train_path + '/'):
        train_image = cv2.cvtColor(cv2.imread(train_path + '/' + serial_number), cv2.COLOR_BGR2GRAY)
        print(serial_number)

        w, h = train_image.shape[1], train_image.shape[0]

        ret = Harris(train_image, 3, 5, 0.04, step_size=1)
        centroids = non_max_suppression(ret, 9)

        for center in centroids:
            x = int(center[0])
            y = int(center[1])
            if x < OFFSET or x >= (w-OFFSET) or y < OFFSET or y >= (h-OFFSET):
                continue
            window = train_image[y-OFFSET:y+1+OFFSET, x-OFFSET:x+1+OFFSET]

            if i == 0:
                patch = (window.flatten()).copy()
                displacement = np.array((x - w / 2, y - h / 2), dtype=np.uint8)
            else:
                patch = np.vstack([patch, window.flatten()])
                displacement = np.vstack([displacement, (x - w / 2, y - h / 2)])

            i += 1

        print(len(centroids))
        # for center in centroids:
        #     cv2.circle(train_image, (int(center[0]), int(center[1])), 2, (255, 0, 0), 1)
        #
        # plt.imshow(train_image, cmap='gray')
        # plt.axis('off')
        # plt.show()

    print(patch.shape)
    print(displacement.shape)
    sio.savemat('patch.mat', {'patch': patch, 'displacement': displacement})


def generateVocabulary():
    M = sio.loadmat('kmeans.mat')
    idx = M['idx']
    C = M['C']

    K = []

    window_size = int(np.sqrt(len(C[0])))

    i = 0
    for frame in C:
        K.append(np.reshape(frame, (window_size, window_size)))
    K = np.array(K)

    train_path = "CarTrainImages"

    OFFSET = int(np.floor(window_size / 2))

    displacements = [list()] * len(K)

    c = 0
    for serial_number in os.listdir(train_path + '/'):
        train_image = cv2.cvtColor(cv2.imread(train_path + '/' + serial_number), cv2.COLOR_BGR2GRAY)
        print(serial_number)

        w, h = train_image.shape[1], train_image.shape[0]

        ret = Harris(train_image, 3, 5, 0.04, step_size=1)
        centroids = non_max_suppression(ret, 9)

        for center in centroids:
            x = int(center[0])
            y = int(center[1])
            if x < OFFSET or x >= (w-OFFSET) or y < OFFSET or y >= (h-OFFSET):
                continue
            SSD = []
            for f in range(len(K)):
                window = train_image[y-OFFSET:y+1+OFFSET, x-OFFSET:x+1+OFFSET]
                ssd = np.mean((window-K[f]) ** 2)
                SSD.append([ssd, f, (x-w/2, y-h/2)])
            SSD.sort(key=lambda z: z[0])

            if len(displacements[SSD[0][1]]) == 0:
                displacements[SSD[0][1]] = [SSD[0][2]]
            else:
                displacements[SSD[0][1]].append(SSD[0][2])

    # for ls in displacements:
    #     ls = list(set(ls))
    #     print(len(ls))

    sio.savemat('displacement.mat', {'d': displacements})


def matchTemplate():
    M = sio.loadmat('displacement.mat')
    displacement = np.squeeze(M['d'])

    M = sio.loadmat('kmeans.mat')
    idx = M['idx']
    C = M['C']
    K = []
    window_size = int(np.sqrt(len(C[0])))
    OFFSET = int(np.floor(window_size / 2))
    for frame in C:
        K.append(np.reshape(frame, (window_size, window_size)))
    K = np.array(K)

    test_path = "CarTestImages"

    c = 0
    for serial_number in os.listdir(test_path + '/'):
        test_image = cv2.cvtColor(cv2.imread(test_path + '/' + serial_number), cv2.COLOR_BGR2GRAY)
        print(serial_number)

        w, h = test_image.shape[1], test_image.shape[0]

        ret = Harris(test_image, 3, 5, 0.04, step_size=1)
        centroids = non_max_suppression(ret, 9)

        output = test_image.copy()
        for center in centroids:
            cv2.circle(output, (int(center[0]), int(center[1])), 2, (255, 0, 0), 1)

        plt.imshow(output, cmap='gray')
        plt.axis('off')
        plt.show()

        GHT_image = np.zeros_like(test_image, dtype=np.uint8)

        for center in centroids:
            x = int(center[0])
            y = int(center[1])
            if x < OFFSET or x >= (w-OFFSET) or y < OFFSET or y >= (h-OFFSET):
                continue
            SSD = []
            for f in range(len(K)):
                window = test_image[y-OFFSET:y+1+OFFSET, x-OFFSET:x+1+OFFSET]
                ssd = np.mean((window-K[f]) ** 2)
                SSD.append([ssd, f, (x-w/2, y-h/2)])
            SSD.sort(key=lambda l: l[0])

            for d in displacement[SSD[0][1]]:
                x_new = int(x - d[0])
                y_new = int(y - d[1])
                if x_new < 0 or x_new >= w or y_new < 0 or y_new >= h:
                    continue
                GHT_image[y_new][x_new] += 1

        centroids = non_max_suppression(GHT_image, window_size=37)

        output = test_image.copy()
        for center in centroids:
            cv2.circle(output, (int(center[0]), int(center[1])), 2, (255, 0, 0), 1)

        GHT_image = cv2.GaussianBlur(GHT_image, (5, 5), 1.5)

        plt.imshow(GHT_image, cmap='gray')
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    # generatePatch()
    # generateVocabulary()
    matchTemplate()
