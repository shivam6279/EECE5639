import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
import os


def estimate_noise(imset):
    avg = np.average(imset, axis=0)

    sigma = np.zeros_like(imset[0])

    for i in range(imset[0].shape[1]):
        for j in range(imset[0].shape[0]):
            for n in range(len(imset)):
                sigma[i][j] += (avg[i][j] - imset[n][i][j]) ** 2

    sigma = (sigma / (len(imgset) - 1)) ** (1/2)

    avg_sigma = np.average(sigma)
    print(avg_sigma)

    return avg_sigma


dataset_path = ["RedChair", "EnterExitCrossingPaths2cor", "Office"]

sigma = 2.0
kernel_size = math.ceil(5 * sigma)
if kernel_size % 2 == 0:
    kernel_size += 1

for path in dataset_path:
    print(path + '/' + path + '/')

    imgset = []
    for serial_number in os.listdir(path + '/' + path + '/'):
        imgset.append(cv2.imread(path + '/' + path + '/' + serial_number))

    # imgset = [cv2.blur(frame, (3, 3)) for frame in imgset]
    # imgset = [cv2.blur(frame, (5, 5)) for frame in imgset]
    imgset = [cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma) for frame in imgset]

    grayscale = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in imgset]

    derivative_set = [np.zeros_like(frame) for frame in grayscale]
    res = imgset.copy()

    for i in range(1, len(imgset) - 1):
        derivative_set[i] = np.fabs((grayscale[i + 1].astype(np.int16) - grayscale[i - 1].astype(np.int16)) / 2).astype(np.uint8)
        _, thresh = cv2.threshold(derivative_set[i], 4, 255, cv2.THRESH_BINARY)

        redImg = np.zeros(imgset[0].shape, imgset[0].dtype)
        redImg[:, :] = (0, 0, 255)

        redMask = cv2.bitwise_and(redImg, redImg, mask=thresh)
        # res[i] = cv2.bitwise_and(cv2.cvtColor(thresh[i], cv2.COLOR_GRAY2BGR), imgset[i])
        cv2.addWeighted(redMask, 1, res[i], 0.75, 0, res[i])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path + ".mp4v", fourcc, 10, (grayscale[0].shape[1], grayscale[0].shape[0]), isColor=0)

    for img in res:
        video.write(img)
