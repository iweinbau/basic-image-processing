"""Skeleton code for python script to process a input using OpenCV package
:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import sys
import numpy as np
import cv2

from utils.video_editor import TimeLineClip, NoEffect, GaussianBlurEffect, BilateralBlurEffect, GreyScaleEffect, \
    HSVMaskEffect, TimeLine, HoughCircleEffect

clips = [TimeLineClip('output/stitched.mp4', 1100, 3320, NoEffect()),
         TimeLineClip('output/stitched.mp4', 5440, 8600, GreyScaleEffect()),
         TimeLineClip('output/stitched.mp4', 9400, 10540, NoEffect()),
         TimeLineClip('output/stitched.mp4', 12080, 12125, GaussianBlurEffect((5, 5))),
         TimeLineClip('output/stitched.mp4', 12125, 13200, GaussianBlurEffect((9, 9))),
         TimeLineClip('output/stitched.mp4', 13200, 14275, GaussianBlurEffect((13, 13))),
         TimeLineClip('output/stitched.mp4', 14275, 15350, GaussianBlurEffect((17, 17))),
         TimeLineClip('output/stitched.mp4', 15350, 16425, BilateralBlurEffect(5, 60)),
         TimeLineClip('output/stitched.mp4', 16425, 17500, BilateralBlurEffect(9, 60)),
         TimeLineClip('output/stitched.mp4', 17500, 18575, BilateralBlurEffect(13, 60)),
         TimeLineClip('output/stitched.mp4', 18575, 19650, BilateralBlurEffect(17, 60)),
         TimeLineClip('output/stitched.mp4', 20510, 21200, NoEffect()),
         TimeLineClip('output/stitched.mp4', 21200, 28810, HSVMaskEffect(np.array([0, 68, 239], np.uint8),
                                                                         np.array([41, 255, 255], np.uint8),
                                                                         iterations=2))]
         # TimeLineClip('input/take_6.mp4', 2320, 7320, NoEffect()),
         # TimeLineClip('input/take_6.mp4', 7320, 7320, HoughCircleEffect())]
         # TimeLineClip('input/take_6.mp4', 17320, 19320, NoEffect()),
         # TimeLineClip('input/take_6.mp4', 19320, 22320, NoEffect())]


def main(output_video_file: str) -> None:
    timeline = TimeLine()
    timeline.add_clips(clips)
    timeline.render_video(output_video_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV input processing')
    parser.add_argument('-i', "--input", help='full path to input input that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed input output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output input files! See --help")

    main(args.output)
