import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy.ma.core import reshape

img_1_tmp = cv2.imread('reference_img1.png', cv2.IMREAD_GRAYSCALE)
img_2 = cv2.imread('reference_img2.png', cv2.IMREAD_GRAYSCALE)

def resize(image, scale_factor: int, up_or_down: str):
    if up_or_down == 'down':
        img_downsized = cv2.pyrDown(image, dstsize=(image.shape[1]//scale_factor, image.shape[0]//scale_factor))
        return img_downsized
    else:
        image_upscaled = cv2.pyrUp(image, dstsize=(image.shape[1]*scale_factor, image.shape[0]*scale_factor))
        return image_upscaled

def stitch_images(image1, image2, H):

    h2, w2 = image2.shape[:2]
    h1, w1 = image1.shape[:2]

    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    all_corners = np.concatenate((warped_corners, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1,1,2)), axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    translation = np.array([[1,0, -xmin], [0,1, -ymin],[0,0,1]])

    panorama = cv2.warpPerspective(image1, translation @ H, (xmax - xmin, ymax - ymin))

    panorama[-ymin:h2 - ymin, -xmin:w2 - xmin] = image2

    return panorama

def SIFT_withFLANN(image_to_align, reference_image, max_features, good_match_percent, return_H=False):

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(image_to_align, None)
    kp2, des2 = sift.detectAndCompute(reference_image, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < good_match_percent * n.distance:
            good.append(m)

    if len(good) > max_features:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if return_H:
            return H

        matchesMask = mask.ravel().tolist()

        image3 = cv2.drawMatches(image_to_align, kp1, reference_image, kp2, good, None, matchColor=(0,255,0), matchesMask=matchesMask, flags=2)
        cv2.imshow('image', image3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Not enough matches are found - {}/{}".format(len(good), max_features))
        matchesMask = None



if __name__ == '__main__':
    img_1 = resize(img_1_tmp, 2, 'down')
    H = SIFT_withFLANN(img_1, img_2,10, 0.7, return_H=True)

    if H is not None:
        panorama = stitch_images(img_1, img_2, H)

        cv2.imshow('panorama', panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()