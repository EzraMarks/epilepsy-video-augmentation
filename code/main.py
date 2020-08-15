import os
import argparse
import cv2
from queue import Queue
from threading import Thread
import processing
import video_augmentation as augment

# performs command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'video',
        type=str,
        help='''Path to the video file for processing.''')
    parser.add_argument(
        '--augmentation',
        default='blend_and_cut',
        choices=['hard_cut', 'blend', 'blend_and_cut', 'contrast_drop',
        'black_out', 'normalize_luminance', 'blend_cut_contrast', 'average_brightness',
        'threshold_brightness', 'normalize_brightness', 'normalize_pixels',
        'average_lab', 'replace_value', 'replace_luminance'],
        help='''Which type of video augmentation to use for flash reduction.''')
    parser.add_argument(
        '--realtime',
        action='store_true',
        help='''Flag for using realtime processing.''')
    return parser.parse_args()

def main():
    # queue to hold frames after reading them in
    input_queue = Queue()
    # queue to hold frames after processing, before displaying them out
    output_queue = Queue()

    # creates video reader
    video = cv2.VideoCapture(ARGS.video)
    if not video.isOpened():
        print("Error Opening Video File")

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    augmentation_func = getattr(augment, ARGS.augmentation)

    # creates video writer, only if --realtime is not enabled
    video_writer = None
    if not ARGS.realtime: video_writer = cv2.VideoWriter('../results/output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
   
    # creates instances of each thread
    reading_thread = processing.ReadingThread(input_queue, video)
    processing_thread = processing.ProcessingThread(input_queue, output_queue, video, augmentation_func)
    writing_thread = processing.WritingThread(output_queue, video, video_writer)

    # start running all threads
    reading_thread.start()
    processing_thread.start()
    writing_thread.start()
    
    # after all threads are finished
    reading_thread.join()
    processing_thread.join()
    writing_thread.join()
    print('Video finished playing')

# makes arguments gloabl
ARGS = parse_args()

main()