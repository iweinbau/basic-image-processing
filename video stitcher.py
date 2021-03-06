"""Skeleton code for python script to process a input using OpenCV package
:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import sys
from os import listdir
from os.path import isfile, join


# helper function to change what you do based on input seconds
from image_processor import ImageProcessor


def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper


def main(input_video_files: str, output_video_file: str, fps=0, width=1280, height=720) -> None:

    # Get the properties of first video file
    cap = cv2.VideoCapture(input_video_files[0])

    if width == 0 and height == 0:
        width = int(cap.get(3))
        height = int(cap.get(4))

    if fps == 0:
        fps = int(round(cap.get(5)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # saving output input as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    for file in input_video_files:
        # OpenCV input objects to work with
        cap = cv2.VideoCapture(file)

        # while loop where the real work happens
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:

                frame = ImageProcessor.rescale_image(frame, width, height)
                out.write(frame)

            # Break the loop
            else:
                break

        # When everything done, release the input capture and writing object
        cap.release()

    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV input processing')
    parser.add_argument('-i', "--input", help='full path of all input files that will be stitched')
    parser.add_argument('-o', "--output", help='full path for saving processed video')
    parser.add_argument('-fps', "--fps", help='Set custom FPS, default is 30')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output input files! See --help")

    if args.fps is None:
        args.fps = 30

    video_files = [args.input + "/" + f for f in listdir(args.input) if isfile(join(args.input, f))
               and f.lower().endswith(('.mp4'))]

    video_files.sort()  # sorts normally by alphabetical order
    video_files.sort(key=len)  # sorts by descending length

    main(video_files, args.output, fps=args.fps, width=720, height=480)
