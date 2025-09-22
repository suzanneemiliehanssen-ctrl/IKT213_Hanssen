import cv2
import os
import numpy as np

img = cv2.imread('lambo.png')

shapes = cv2.imread('shapes-1.png')
shapes_template = cv2.imread('shapes_template.jpg')


def save_image(file_name, image):

    directory = r'solutions'

    os.chdir(directory)

    cv2.imwrite(file_name, image)


def sobel_edge_detection(image):

    img_blur = cv2.GaussianBlur(image, (3, 3), 0)
    sobel_img = cv2.Sobel(img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)

    cv2.imshow('Sobel Edge Detection', sobel_img)

    file_name = 'sobel_edge_detection.png'
    save_image(file_name, sobel_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def canny_edge_detection(image, threshold_1, threshold_2):
    img_blur = cv2.GaussianBlur(image, (3, 3), 0)
    canny_img = cv2.Canny(img_blur, threshold_1, threshold_2)

    cv2.imshow('Canny Edge Detection', canny_img)

    file_name = 'canny_edge_detection.png'
    save_image(file_name, canny_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def template_match(image, template):
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    matchTemplate = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)


    threshold = 0.9
    loc = np.where(matchTemplate >= threshold)
    for pt in zip(*loc[::-1]):
        res = cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv2.imshow('Match Template', res)

    file_name = 'template_match.png'
    save_image(file_name, res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(image, scale_factor: int, up_or_down: str):
    if up_or_down == 'down':
        img_downsized = cv2.pyrDown(image, dstsize=(image.shape[1]//scale_factor, image.shape[0]//scale_factor))
        cv2.imshow('Downsized Image', img_downsized)

        file_name = 'downsized_image.png'
        save_image(file_name, img_downsized)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        image_upscaled = cv2.pyrUp(image, dstsize=(image.shape[1]*scale_factor, image.shape[0]*scale_factor))
        cv2.imshow('Upscaled Image', image_upscaled)

        file_name = 'upscaled_image.png'
        save_image(file_name, image_upscaled)

        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    sobel_edge_detection(img)
    canny_edge_detection(img, 50, 50)
    template_match(shapes, shapes_template)
    resize(img, 2, 'up')
    resize(img, 2, 'down')

