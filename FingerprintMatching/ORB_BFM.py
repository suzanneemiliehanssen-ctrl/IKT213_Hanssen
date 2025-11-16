import numpy as np
import pip
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


def ORB_feature_matching(image1, image2):
    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    #ORB detector

    orb = cv2.ORB_create(nfeatures=1000)

    #find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    #Create BFMatcher object with Hamming (suitable for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    #Match descriptors
    matches_total = bf.match(des1, des2)

    #Filter out the best matches
    matches = sorted(matches_total, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    output_img = resize(img_matches, 2, 'down')
    cv2.imshow('Feature Matches', output_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    ORB_feature_matching(img_1, img_2)

