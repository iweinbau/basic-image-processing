from abc import ABC, abstractmethod
import cv2
from image_processor import ImageProcessor
import numpy as np


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
        return image


class GreyScaleEffect(Effect):
    """
    Turn image into greyscale
    """

    def __init__(self):
        super().__init__("GreyScale")

    def apply_effect(self, image):
        return ImageProcessor.bgr_to_gray(image)


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
        return ImageProcessor.gaussian_blur(image, self.kernel, self.sigma_x, self.sigma_y)


class BilateralBlurEffect(Effect):
    """
    Apply bilateral blur effect.
    """

    def __init__(self, color_space, color_range):
        super().__init__("BlurEffect")
        self.color_space = color_space
        self.color_range = color_range

    def apply_effect(self, image):
        return ImageProcessor.bilateral_filer(image, self.color_space, self.color_range)


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

        return result


class SobelEffect(Effect):
    def __init__(self, dx, dy, kernel_size, scale=1, delta=0):
        super().__init__("Sobel")
        self.dx = dx
        self.dy = dy
        self.kernel_size = kernel_size
        self.scale = scale
        self.delta = delta

    def apply_effect(self, image):
        gray = ImageProcessor.bgr_to_gray(image)
        return cv2.Sobel(gray, -1, self.dx, self.dy, ksize=self.kernel_size, scale=self.scale, delta=self.delta)


class HoughCircleEffect(Effect):
    def __init__(self):
        super().__init__("Hough Circles")

    def apply_effect(self, image):
        blur = cv2.blur(image, (7,7))
        gray = ImageProcessor.bgr_to_gray(blur)
        edge_v = cv2.Sobel(gray, -1, 1, 0, ksize=3, scale=1, delta=-50)
        edge_h = cv2.Sobel(gray, -1, 0, 1, ksize=3, scale=1, delta=-50)
        edge = edge_v + edge_h


        return cv2.Canny(edge, 200, 10)
