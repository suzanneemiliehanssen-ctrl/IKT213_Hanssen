import cv2
import os
import numpy as np

lena = cv2.imread('lena-1.png')

def print_image(image):
    cv2.imshow('my_image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(file_name, image):

    directory = r'solutions'

    os.chdir(directory)

    cv2.imwrite(file_name, image)


#Create a border around the original image which reflects the edges of the original image
def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    cv2.imshow('padded_image', padded_image)

    file_name = 'padded_image.png'

    save_image(file_name, padded_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Cut the image such that it returns only the part of the image which is of interest
# (in your case, the face on lena.png)
def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[80:382, 80:382]
    cv2.imshow('cropped_image', cropped_image)

    file_name = 'cropped_image.png'
    save_image(file_name, cropped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Create a function which lets you resize the image
def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imshow('resized_image', resized_image)

    file_name = 'resized_image.png'
    save_image(file_name, resized_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def copy(image):
    height, width, channels = image.shape
    empty_picture_array = np.zeros((height, width, channels), dtype=np.uint8)
    copied_image = image + empty_picture_array

    cv2.imshow('copied_image', copied_image)

    file_name = 'copied_image.png'
    save_image(file_name, copied_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



#Convert the colored image to a grayscale image
def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayscale', gray_image)

    file_name = 'grayscale.png'
    save_image(file_name, gray_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



#Convert an RGB image to use HSV
def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv_image', hsv_image)

    file_name = 'hsv_image.png'
    save_image(file_name, hsv_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Shift the color values of a given RGB image
def hue_shifted(image, emptyPictureArray, hue):

    for i in range(len(image)):
        for j in range(len(image[i])):
            r = image[i][j][0]
            g = image[i][j][1]
            b = image[i][j][2]

            if 255 - hue <= r <= 255:
                new_r = 255
            else: new_r = r + hue

            if 255 - hue <= g <= 255:
                new_g = 255
            else: new_g = g + hue

            if 255 - hue <= b <= 255:
                new_b = 255
            else: new_b = b + hue

            emptyPictureArray[i][j] = [new_r, new_g, new_b]

    cv2.imshow('hue_shifted', emptyPictureArray)

    file_name = 'hue_shifted.png'
    save_image(file_name, emptyPictureArray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



#You will be smoothing the original image; adding a blur to the image
def smoothing(image):
    smooth_image = cv2.GaussianBlur(image, (15, 15), 0, borderType=cv2.BORDER_DEFAULT)
    cv2.imshow('smooth_image', smooth_image)

    file_name = 'smooth_image.png'
    save_image(file_name, smooth_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotation(image, rotation_angle):
    if rotation_angle == "90":
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        file_name = 'rotated_90_image.png'
        save_image(file_name, rotated_image)

    if rotation_angle == "180":
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        file_name = 'rotated_180_image.png'
        save_image(file_name, rotated_image)

    cv2.imshow('rotated_image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

def main():
    #padding(lena, 100) #functional


    #crop(lena, 50, 50, 50, 50) #functional


    #resize(lena, 200, 200) #functional


    #copy(lena) #functional


    #grayscale(lena) #functional


    #hsv(lena) #functional


    #shifted_img_array = np.zeros_like(lena, dtype=np.uint8)
    #hue_shifted(lena, shifted_img_array, 50) #functional


    #smoothing(lena) #functional


    user_input = input("Input '90' to rotate image 90 degrees clockwise"
                       "or input '180' to rotate image 180 degrees.")
    rotation(lena, user_input) #functional



main()
