import cv2
import numpy as np

img_1 = cv2.imread('UiA_front_1.jpg')
img_2 = cv2.imread('UiA_front_2.png')

def resize(image, scale_factor: int, up_or_down: str):
    if up_or_down == 'down':
        img_downsized = cv2.pyrDown(image, dstsize=(image.shape[1]//scale_factor, image.shape[0]//scale_factor))
        return img_downsized
    else:
        image_upscaled = cv2.pyrUp(image, dstsize=(image.shape[1]*scale_factor, image.shape[0]*scale_factor))
        return image_upscaled


def SIFT_feature_matching(img_1, img_2):
    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    #initialize SIFT
    sift = cv2.SIFT_create()

    #Find keypoint and descriptors
    kp1, des1 = sift.detectAndCompute(img_1_gray, None)
    kp2, des2 = sift.detectAndCompute(img_2_gray, None)

    #Create BFMatcher object with L2 distance (suitable for SIFT floating-point des)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    #knn
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    img_matches = cv2.drawMatches(img_1, kp1, img_2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    output_img = resize(img_matches, 2, 'down')
    cv2.imshow('Feature Matches', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    SIFT_feature_matching(img_1, img_2)