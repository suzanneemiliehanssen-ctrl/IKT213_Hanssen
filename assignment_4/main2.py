import cv2
import numpy as np

def harris_corner_detection(reference_image):
    img = cv2.imread('harris.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    dst = cv2.dilate(dst,None)

    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imwrite("test.png",img)