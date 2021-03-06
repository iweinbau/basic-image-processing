import cv2
import numpy as np


class ImageProcessor:
    @staticmethod
    def rescale_image(image, width, height):
        dim = (width, height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def bgr_to_gray(image, multi_channel=True):
        """
        Convert image in BRG format to gray scale
        :param image: original input image
        :param multi_channel: True if it has to output it in RGB channels, False for single channel output.
        Default is set to True
        :return: new image in gray scale
        """
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if multi_channel:
            gray_scale_channels = cv2.cvtColor(gray_scale, cv2.COLOR_GRAY2BGR)
            gray_scale = gray_scale_channels

        return gray_scale

    @staticmethod
    def bgr_to_hsv(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    @staticmethod
    def hsv_to_bgr(image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    @staticmethod
    def gaussian_blur(image, kernel_size, sigma_x=0, sigma_y=0):
        return cv2.GaussianBlur(image, kernel_size, sigmaX=sigma_x, sigmaY=sigma_y)

    @staticmethod
    def bilateral_filer(image, sigma_space, sigma_range):
        return cv2.bilateralFilter(image, -1, sigma_range, sigma_space)

    @staticmethod
    def mask_hsv(hsv_image, lower_bound, upper_bound):
        return cv2.inRange(hsv_image, lower_bound, upper_bound)  # Thresholding



