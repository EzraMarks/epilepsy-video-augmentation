import cv2
import numpy as np
from queue import Queue
from threading import Thread
from progress.bar import IncrementalBar

from flash_detection import detect_flashes


class ProgressBar(IncrementalBar):
    suffix = "%(percent)d%% [%(elapsed_td)s / %(eta_td)s]"


class ReadingThread():
    def __init__(self, input_queue, original_queue, video):
        self.name = "Reading Thread"
        self.input_queue = input_queue
        self.original_queue = original_queue
        self.video = video
        self.run()

    def run(self):
        bar = ProgressBar("Reading", max=self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        go = self.video.isOpened()
        while (go):
            # load frames into queue
            go, frame = self.video.read()
            if (go):
                self.input_queue.put(frame)
                self.original_queue.put(frame)
            bar.next()
        bar.finish()
        self.video.release()


class ProcessingThread():
    def __init__(self, input_queue, output_queue, video, augmentation_func):
        self.name = "Processing Thread"
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.video = video
        self.augmentation_func = augmentation_func
        self.run()

    def run(self):
        bar = ProgressBar("Processing", max=self.input_queue.qsize())
        frame_w  = self.input_queue.queue[0].shape[1]
        frame_h = self.input_queue.queue[0].shape[0]
        num_frames = 10
        overlap = 0
        frames = np.zeros((frame_h, frame_w, 3, num_frames), dtype=np.uint8)
        total_after = 0
        total = 0

        while (self.input_queue.qsize() >= num_frames):
            # loading frames into array for processing:
            for i in range(num_frames - overlap):
                bar.next()
                # pop these frames off of the input queue,
                # since we will no longer need them
                frames[:, :, :, i] = self.input_queue.get()
            for i in range(overlap):
                # keep these frames in the input queue for reuse (overlap)
                # when processing the next segment of the video
                frames[:, :, :, num_frames - overlap +
                       i] = self.input_queue.queue[i]

            flash = detect_flashes(frames)
            
            if flash:
                frames = self.augmentation_func(frames)
                total = total + 1
                if detect_flashes(frames):
                    total_after = total_after + 1
            for i in range(num_frames - overlap):
                frame = np.copy(frames[:, :, :, i])
                self.output_queue.put(frame)

        while not self.input_queue.empty():
            bar.next()
            self.output_queue.put(self.input_queue.get())

        bar.finish()
        if total > 0:
            print((total_after / total) * 100)
        print(total)


class WritingThread():
    def __init__(self, output_queue, original_queue, fps):
        self.name = "Writing Thread"
        self.output_queue = output_queue
        self.original_queue = original_queue
        self.fps = fps
        self.run()

    def run(self):
        while self.original_queue.qsize() > 0:
            # read frames from queue
            frame = self.output_queue.get()
            cv2.imshow('frame', frame)
            cv2.waitKey(int(1000 / fps))

        cv2.destroyAllWindows()