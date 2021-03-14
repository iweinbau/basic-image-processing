import textwrap
from abc import ABC, abstractmethod
import cv2
from image_processor import ImageProcessor
import numpy as np


TEXT_FOND =cv2.FONT_HERSHEY_SIMPLEX

class TimeLine:
    def __init__(self):
        self.clips = []

    def add_clip(self, clip):
        self.clips.append(clip)

    def insert_clip(self, clip, index):
        self.clips.insert(clip, index)

    def add_clips(self, new_clips):
        self.clips.extend(new_clips)

    def render_video(self, output_file, fps=0, width=0, height=0, should_display=True):

        if not self.clips:
            print("Time line is empty")
            return

        cap = cv2.VideoCapture(self.clips[0].source)
        if width == 0 and height == 0:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps == 0:
            fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # saving output input as .mp4
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        for clip in self.clips:
            # OpenCV input objects to work with
            cap = cv2.VideoCapture(clip.source)
            cap.set(cv2.CAP_PROP_POS_MSEC, clip.start)

            # while loop where the real work happens
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:

                    if cap.get(cv2.CAP_PROP_POS_MSEC) > clip.end:
                        break

                    if cv2.waitKey(28) & 0xFF == ord('q'):
                        break

                    # frame = ImageProcessor.rescale_image(frame, width, height)
                    frame = clip.apply_effect(frame)

                    # write frame that you processed to output
                    out.write(frame)

                    if should_display:
                        # (optional) display the resulting frame
                        cv2.imshow('Frame', frame)

                # Break the loop
                else:
                    break

            cap.release()

        out.release()
        # Closes all the frames
        cv2.destroyAllWindows()


class TimeLineClip:
    """
    Represents a clip on the time line, has a start en end value from the original video
    And has an clip effect.
    """

    def __init__(self, source, start, end, effect):
        self.source = source
        self.start = start
        self.end = end
        self.effect = effect

    def apply_effect(self, image):
        return self.effect.apply_effect(image)


class Effect(ABC):
    """
    Class to represent an effect on a clib. Effects can be changed.
    """

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def apply_effect(self, image):
        """
        Apply special effect to image
        :param image: image where to apply the effect on
        """
        pass


class NoEffect(Effect):
    """
    Apply no effect, Use the Raw input
    """

    def __init__(self):
        super().__init__("Raw")

    def apply_effect(self, image):
        wrapped_text = textwrap.wrap('Raw input', width=35)
        for i, line in enumerate(wrapped_text):
            textsize = cv2.getTextSize(line, TEXT_FOND, 1, 2)[0]
            gap = textsize[1] + 10
            x = 10
            y = 50 + i*gap
            cv2.putText(image, line, (x, y), TEXT_FOND, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return image


class GreyScaleEffect(Effect):
    """
    Turn image into greyscale
    """

    def __init__(self):
        super().__init__("GreyScale")

    def apply_effect(self, image):
        gray = ImageProcessor.bgr_to_gray(image)
        wrapped_text = textwrap.wrap('Gray scale', width=35)
        for i, line in enumerate(wrapped_text):
            textsize = cv2.getTextSize(line, TEXT_FOND, 1, 2)[0]
            gap = textsize[1] + 10
            x = 10
            y = 50 + i*gap
            cv2.putText(gray, line, (x, y), TEXT_FOND, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return gray


class GaussianBlurEffect(Effect):
    """
    Apply guassian blur effect. The first
    """

    def __init__(self, kernel, sigma_x=0, sigma_y=0):
        super().__init__("BlurEffect")
        self.kernel = kernel
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def apply_effect(self, image):
        blur = ImageProcessor.gaussian_blur(image, self.kernel, self.sigma_x, self.sigma_y)
        wrapped_text = textwrap.wrap('Gaussian blur: {} kernel'.format(self.kernel), width=35)
        for i, line in enumerate(wrapped_text):
            textsize = cv2.getTextSize(line, TEXT_FOND, 1, 2)[0]
            gap = textsize[1] + 10
            x = 10
            y = 50 + i*gap
            cv2.putText(blur, line, (x, y), TEXT_FOND, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return blur


class BilateralBlurEffect(Effect):
    """
    Apply bilateral blur effect.
    """

    def __init__(self,sigma_space, sigma_range):
        super().__init__("BlurEffect")
        self.sigma_range = sigma_range
        self.sigma_space = sigma_space

    def apply_effect(self, image):
        blur = cv2.bilateralFilter(image, -1, self.sigma_range, self.sigma_space)
        wrapped_text = textwrap.wrap('Bilateral filter: sigmaColor={}, sigmaRange={}'
                    .format(self.sigma_range, self.sigma_space), width=35)
        for i, line in enumerate(wrapped_text):
            textsize = cv2.getTextSize(line, TEXT_FOND, 1, 2)[0]
            gap = textsize[1] + 10
            x = 10
            y = 50 + i*gap
            cv2.putText(blur, line, (x, y), TEXT_FOND, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return blur


class HSVMaskEffect(Effect):

    def __init__(self, lower_bound, upper_bound, iterations=0):
        super().__init__("HSVMask")
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.iterations = iterations

    def apply_effect(self, image):
        # blur image first to get better mask
        blur = ImageProcessor.gaussian_blur(image, (7, 7), sigma_x=1.5, sigma_y=1.5)
        mask = ImageProcessor.mask_hsv(ImageProcessor.bgr_to_hsv(blur), self.lower_bound, self.upper_bound)

        if self.iterations > 0:
            kernel = np.ones((5, 5))
            erosion = cv2.erode(mask, kernel, iterations=self.iterations)

            erosion_diff = mask - erosion

            dilation = cv2.dilate(erosion, kernel, iterations=self.iterations)

            dilation_diff = dilation - erosion

            # Convert mask to 3 channel BGR
            result = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)

            # Color erosion red and dilation green
            result[(erosion_diff == 255)] = [0, 0, 255]
            result[(dilation_diff == 255)] = [0, 255, 0]
        else:
            # Convert mask to 3 channel BGR
            result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        wrapped_text = textwrap.wrap('HSV-mask: Red is removed with erosion, Green is added back with dilation', width=35)
        for i, line in enumerate(wrapped_text):
            textsize = cv2.getTextSize(line, TEXT_FOND, 1, 2)[0]
            gap = textsize[1] + 10
            x = 10
            y = 50 + i*gap
            cv2.putText(result, line, (x, y), TEXT_FOND, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return result


class SobelEffect(Effect):
    def __init__(self, kernel_size=3, scale=1, delta=0):
        super().__init__("Sobel")
        self.kernel_size = kernel_size
        self.scale = scale
        self.delta = delta

    def apply_effect(self, image):
        gray = ImageProcessor.bgr_to_gray(image, multi_channel=False)
        blur = cv2.GaussianBlur(gray, (5, 5), 0, 0)
        horizontal = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=self.kernel_size, scale=self.scale, delta=self.delta)
        vertical = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=self.kernel_size, scale=self.scale, delta=self.delta)

        horizontal = cv2.convertScaleAbs(horizontal)
        vertical = cv2.convertScaleAbs(vertical)

        color_edge = np.zeros_like(image)
        color_edge[:, :, 1] = horizontal
        color_edge[:, :, 2] = vertical

        wrapped_text = textwrap.wrap('Sobel edge detection ({},{}) kernel: vertical edges red, horizontal edges green'
                    .format(self.kernel_size, self.kernel_size), width=35)
        for i, line in enumerate(wrapped_text):
            textsize = cv2.getTextSize(line, TEXT_FOND, 1, 2)[0]
            gap = textsize[1] + 10
            x = 10
            y = 50 + i*gap
            cv2.putText(color_edge, line, (x, y), TEXT_FOND, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return color_edge


class HoughCircleEffect(Effect):
    def __init__(self, dp=1, min_dist=200, param1=200, param2=45):
        super().__init__("Hough Circles")
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2

    def apply_effect(self, image):
        gray = ImageProcessor.bgr_to_gray(image, multi_channel=False)
        blur = cv2.GaussianBlur(gray, (5, 5), 0, 0)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 200,
                                   param1=200, param2=42, minRadius=50, maxRadius=200)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

        wrapped_text = textwrap.wrap('HoughCircle: dp={}, Canny upper threshold={}, Hough threshold={}'
                                     .format(self.dp, self.param1, self.param2), width=35)
        for i, line in enumerate(wrapped_text):
            textsize = cv2.getTextSize(line, TEXT_FOND, 1, 2)[0]
            gap = textsize[1] + 10
            x = 10
            y = 50 + i*gap
            cv2.putText(image, line, (x, y), TEXT_FOND, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return image


class FlashyBoxEffect(Effect):
    def __init__(self, lower_bound, upper_bound):
        super().__init__("Flashy Rectangle")
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def apply_effect(self, image):
        # blur image first to get better mask
        blur = ImageProcessor.gaussian_blur(image, (7, 7), sigma_x=1.5, sigma_y=1.5)
        mask = ImageProcessor.mask_hsv(ImageProcessor.bgr_to_hsv(blur), self.lower_bound, self.upper_bound)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        has_contour = False
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            area = w*h
            if max_area < area:
                max_area = area
                max_contour = c
                has_contour = True
        if has_contour:
            # draw a green rectangle to visualize the bounding rect
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        wrapped_text = textwrap.wrap('Object of interest', width=35)
        for i, line in enumerate(wrapped_text):
            textsize = cv2.getTextSize(line, TEXT_FOND, 1, 2)[0]
            gap = textsize[1] + 10
            x = 10
            y = 50 + i*gap
            cv2.putText(image, line, (x, y), TEXT_FOND, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return image


class TemplateMatchEffect(Effect):
    def __init__(self):
        super().__init__("Template match")

    def apply_effect(self, image):
        gray = ImageProcessor.bgr_to_gray(image, multi_channel=False)
        blur = cv2.GaussianBlur(gray, (5, 5), 0, 0)

        # Find circle
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 200,
                                   param1=200, param2=42, minRadius=50, maxRadius=200)

        # Extract circle out of image to use as template
        if circles is not None:
            circles = np.uint16(np.around(circles))
            circle = circles[0, 0]
            circle_center_x = circle[0]
            circle_center_y = circle[1]
            circle_radius = circle[2]

            pattern = image[circle_center_y-circle_radius:circle_center_y+circle_radius,
                            circle_center_x - circle_radius:circle_center_x + circle_radius,:]

            res = cv2.matchTemplate(image, pattern, cv2.TM_SQDIFF_NORMED)
            height, width, _ = image.shape
            res = cv2.resize(res, (width, height), interpolation=cv2.INTER_AREA)
            return res

        return image
