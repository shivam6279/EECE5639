import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
import os


# Find the noise estimate of a set of images
def estimate_noise(imset):
    avg = np.average(imset, axis=0)
    print(avg.shape)
    print(imset[0].shape)
    sig = np.zeros_like(imset[0], dtype=np.float64)

    for n in range(len(imset)):
        sig += (avg - imset[n].astype(dtype=np.float64)) ** 2

    sig = (sig  / (len(imgset) - 1)) ** (1/2)

    avg_sigma = np.average(sig)
    print(avg_sigma)

    return avg_sigma


# Computes the 1-D derivative of a gaussian filter given an input sigma
# Normalizes the filter so that the sum of the magnitude of each element equals 1
def get_gaussian_derivative(sdev):
    filter_size = math.ceil(sigma * 5)

    if filter_size % 2 == 0:
        filter_size += 1

    filter = 0.5 * np.empty(filter_size)
    mid = math.floor(filter_size / 2)

    for i in range(filter_size):
        filter[i] = -(i - mid) * math.exp((-(i - mid) ** 2) / (2 * sdev ** 2))

    filter /= sum(np.fabs(filter))
    return filter


dataset_path = ["RedChair", "EnterExitCrossingPaths2cor", "Office"]

sigma = 2.0

# Calculate the kernel size based on sigma
kernel_size = math.ceil(5 * sigma)
if kernel_size % 2 == 0:
    kernel_size += 1

for path in dataset_path:
    print(path + '/' + path + '/')

    # Open each image in the specified folder and save them in imgset
    imgset = []
    for serial_number in os.listdir(path + '/' + path + '/'):
        imgset.append(cv2.imread(path + '/' + path + '/' + serial_number))

    # Blur all the images in imgset

    # imgset = [cv2.blur(frame, (3, 3)) for frame in imgset]
    # imgset = [cv2.blur(frame, (5, 5)) for frame in imgset]
    imgset = [cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma) for frame in imgset]

    # Convert all the images to grayscale
    grayscale = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in imgset]
    estimate_noise(grayscale)
    derivative_set = [np.zeros_like(frame, dtype=np.float32) for frame in grayscale]
    res = imgset.copy()

    #kernel = 0.5 * np.array([-1, 0, 1])
    kernel = get_gaussian_derivative(5.0)

    kernel_mid = math.floor((len(kernel) / 2))
    for i in range(kernel_mid, len(imgset) - kernel_mid):
        # Calculate the derivative based on the filter
        # derivative_set[i] = np.fabs((grayscale[i + 1].astype(np.float32) - grayscale[i - 1].astype(np.float32)) / 2.0)
        for k in range(len(kernel)):
            derivative_set[i] += grayscale[i + k - kernel_mid].astype(np.float32) * kernel[k]
        derivative_set[i] = np.fabs(derivative_set[i])

        # Threshold the derivative
        _, thresh = cv2.threshold(derivative_set[i], 6.0, 255, cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)

        # Create a red image to mask the derivative onto the original image
        redImg = np.zeros(imgset[0].shape, imgset[0].dtype)
        redImg[:, :] = (0, 0, 255)
        redMask = cv2.bitwise_and(redImg, redImg, mask=thresh)

        # Add the red mask with the original image
        cv2.addWeighted(redMask, 1, res[i], 0.75, 0, res[i])

    # Save the set of images as a .mp4v video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path + ".mp4v", fourcc, 10, (grayscale[0].shape[1], grayscale[0].shape[0]), isColor=0)
    for img in res:
        video.write(img)
