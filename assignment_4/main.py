import cv2
import numpy as np

def harris_detection(image):
    image = cv2.imread("align_this.jpg")
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dst = cv2.cornerHarris(image_gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(image_gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    res = np.hstack((centroids, corners))
    res = np.int8(res)
    image[res[:,1], res[:,0]]=[0,0,255]
    image[res[:,3], res[:,2]]=[0,255,0]

    cv2.imwrite("test2.jpg", image)
