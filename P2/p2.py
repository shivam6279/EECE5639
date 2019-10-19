import numpy as np 
from matplotlib import pyplot as plt 
import math 
import cv2
import os 
import gc



dataset_path = ["DanaHallWay1"]#, "DanaHallWay2", "DanaOffice"]


for path in dataset_path:

    imgset = []
    for serial_number in os.listdir(path + "/"):
        if (len(imgset) < 2):
            imgset.append(cv2.imread(path + "/" + serial_number))

    grayscale = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)for img in imgset]
    grayscale = np.float32(grayscale)

    dst = [cv2.cornerHarris(img,2,3,0.4) for img in grayscale]
    
    for img in dst:
        cv2.imshow("dst", img)

        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
