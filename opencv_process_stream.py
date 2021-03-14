import cv2
import numpy as np

from image_processor import ImageProcessor
from utils.video_editor import *


def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper


def main() -> None:
    # OpenCV input objects to work with
    cap = cv2.VideoCapture(0)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output input as .mp4
    # out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            if between(cap, 0, 500):
                # do something using OpenCV functions (skipped here so we simply write the input frame back to output)
                pass
            # ...

            effect_1 = TemplateMatchEffect()

            frame_ = effect_1.apply_effect(frame)
            # (optional) display the resulting frame
            cv2.imshow('Frame', frame_)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if cv2.waitKey(25) & 0xFF == ord('s'):
                cv2.imwrite("output.png", frame)
        # Break the loop
        else:
            break

    # When everything done, release the input capture and writing object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main();