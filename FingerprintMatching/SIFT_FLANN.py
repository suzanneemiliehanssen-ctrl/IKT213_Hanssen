import numpy as np
import cv2

img_1 = cv2.imread('UiA_front_1.jpg')
img_2 = cv2.imread('UiA_front_2.png')

def resize(image, scale_factor: int, up_or_down: str):
    if up_or_down == 'down':
        img_downsized = cv2.pyrDown(image, dstsize=(image.shape[1]//scale_factor, image.shape[0]//scale_factor))
        return img_downsized
    else:
        image_upscaled = cv2.pyrUp(image, dstsize=(image.shape[1]*scale_factor, image.shape[0]*scale_factor))
        return image_upscaled

def ORB_feature_matching(img_1, img_2):
    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img_1_gray, None)
    kp2, des2 = orb.detectAndCompute(img_2_gray, None)

    if des1 is None or des2 is None:
        raise ValueError("Could not find descriptors.")

    FLANN_INDEX_KDTREE = 1
    #FLANN parameters
    index_params = dict(algorithm=cv2.FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=5)
