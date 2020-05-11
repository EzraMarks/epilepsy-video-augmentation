import os
import argparse
import cv2
from queue import Queue
from threading import Thread
import realtime_processing as realtime
import non_realtime_processing as non_realtime

exec(open("./video_augmentation.py").read())

# perform command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'video',
        type=str,
        help='''Path to the video file for processing.''')
    parser.add_argument(
        '--augmentation',
        default='blend',
        choices=['blend', 'contrast_drop'], #TODO
        help='''Which type of video augmentation to use for flash reduction.''')
    parser.add_argument(
        '--realtime',
        action='store_true',
        help='''Flag for using realtime processing.''')
    return parser.parse_args()

def main():
    # FIFO (first-in-first-out) queue to hold frames after reading them in
    input_queue = Queue()
    # FIFO queue to hold frames after processing, before writing/displaying them out
    output_queue = Queue()

    # create video reader
    video = cv2.VideoCapture(ARGS.video)
    if not video.isOpened():
        print("Error Opening Video File")
    fps = video.get(cv2.CAP_PROP_FPS)

    augmentation_func = globals()[ARGS.augmentation]

    if (ARGS.realtime):
        # create instances of each thread
        reading_thread = realtime.ReadingThread(input_queue, video)
        processing_thread = realtime.ProcessingThread(input_queue, output_queue, video, augmentation_func)
        writing_thread = realtime.WritingThread(output_queue, video)

        # start running all threads
        reading_thread.start()
        processing_thread.start()
        writing_thread.start()
    else:
        # create queue to store unaltered video for side-by-side comparison
        original_queue = Queue()
        # create instances of each thread
        reading_thread = non_realtime.ReadingThread(input_queue, original_queue, video)
        processing_thread = non_realtime.ProcessingThread(input_queue, output_queue, video, augmentation_func)
        writing_thread = non_realtime.WritingThread(output_queue, original_queue, fps)
    

    # after all threads are finished
    reading_thread.join()
    processing_thread.join()
    writing_thread.join()
    print('Video finished playing')

# make arguments gloabl
ARGS = parse_args()

main()