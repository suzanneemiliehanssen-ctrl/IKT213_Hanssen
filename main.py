import cv2

lena = cv2.imread('lena-1.png')

def print_image_information(image):
    height, width, channels = image.shape
    print('Image height: ', height, '\nImage width: ', width, '\nImage channels: ', channels)

    size = image.size
    d_type = image.dtype

    print('Image size: ', size, '\nImage data type: ', d_type)


def main():
    print_image_information(lena)

main()